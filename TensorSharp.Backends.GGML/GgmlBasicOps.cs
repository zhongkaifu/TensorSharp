// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp.Core;
using TensorSharp.Cpu;

namespace TensorSharp.GGML
{
    // Per-sequence descriptor for the batched Qwen3.5 GDN native step. Mirrors
    // GdnBatchedSeqDesc in ggml_ops_gated_delta_net.cpp; layout must stay in
    // sync with the native struct (32 bytes on 64-bit).
    [StructLayout(LayoutKind.Sequential)]
    public struct GdnBatchedSeqDesc
    {
        public int   SeqStart;
        public int   SeqLen;
        public int   ConvWriteIdx;  // in/out
        public int   Pad;
        public IntPtr ConvState;    // float* — [(convKernel-1) * qkvDim]
        public IntPtr SsmState;     // float* — [numVHeads * headVDim * headKDim]
    }

    // Per-sequence descriptor for the batched Nemotron Mamba2 native step.
    // Mirrors NemoMamba2BatchedSeqDesc in ggml_ops_mamba2.cpp; 32 bytes on 64-bit.
    // No write-index field because the Mamba2 conv state is a FIFO (memmove
    // by one row per token), not a ring buffer.
    [StructLayout(LayoutKind.Sequential)]
    public struct NemoMamba2BatchedSeqDesc
    {
        public int    SeqStart;
        public int    SeqLen;
        public int    Pad0;
        public int    Pad1;
        public IntPtr ConvState;   // float* — [(dConv-1) * xbcSize]
        public IntPtr SsmState;    // float* — [nHead * headDim * dState]
    }

    [OpsClass]
    public class GgmlBasicOps
    {
        [RegisterOpStorageType("fill", typeof(GgmlStorage))]
        public static unsafe void Fill(Tensor result, float value)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            if (!(result.Storage is GgmlStorage))
                throw new ArgumentException("Fill requires a GGML tensor.", nameof(result));

            // Float16 KV-cache initialization path. The F16 KV cache is always allocated
            // contiguous (head, seqLen, headDim) by the model, so we fill it as a flat
            // ushort[]. value=0 is the common reset case which becomes a plain memset.
            if (result.ElementType == DType.Float16)
            {
                if (!IsContiguousNonNarrowed(result))
                    throw new NotSupportedException("Fill on Float16 GGML tensors requires a contiguous, non-narrowed tensor (used for KV-cache reset).");
                ushort halfBits = System.BitConverter.HalfToUInt16Bits((System.Half)value);
                ushort* halfBuffer = (ushort*)GetBufferStart(result);
                long elementCount = result.ElementCount();
                if (halfBits == 0)
                {
                    // memset for the common cache-reset case.
                    new Span<ushort>(halfBuffer, checked((int)Math.Min(elementCount, int.MaxValue))).Clear();
                    long remaining = elementCount - int.MaxValue;
                    if (remaining > 0)
                    {
                        long offset = int.MaxValue;
                        while (remaining > 0)
                        {
                            int slice = (int)Math.Min(remaining, int.MaxValue);
                            new Span<ushort>(halfBuffer + offset, slice).Clear();
                            offset += slice;
                            remaining -= slice;
                        }
                    }
                }
                else
                {
                    for (long i = 0; i < elementCount; i++)
                        halfBuffer[i] = halfBits;
                }
                return;
            }

            // Q8_0 KV-cache initialization path. Q8_0 stores 32-element blocks of
            // 34 bytes each (16-bit scale + 32 int8 values); the all-zero block
            // (scale=0, quants=0) decodes to all zeros, so a plain memset of the
            // entire buffer is the correct fill-with-0 implementation. Non-zero
            // fills aren't currently used for Q8_0 caches (callers only zero them
            // at session reset).
            if (result.ElementType == DType.Q8_0)
            {
                if (!IsContiguousNonNarrowed(result))
                    throw new NotSupportedException("Fill on Q8_0 GGML tensors requires a contiguous, non-narrowed tensor (used for KV-cache reset).");
                if (value != 0f)
                    throw new NotSupportedException("Fill on Q8_0 GGML tensors only supports value=0 (cache reset).");
                long byteLength = DTypeExtensions.Q8_0Bytes(result.ElementCount());
                byte* byteBuffer = (byte*)GetBufferStart(result);
                long offset = 0;
                while (offset < byteLength)
                {
                    int slice = (int)Math.Min(byteLength - offset, int.MaxValue);
                    new Span<byte>(byteBuffer + offset, slice).Clear();
                    offset += slice;
                }
                return;
            }

            if (result.ElementType != DType.Float32)
                throw new InvalidOperationException($"Fill only supports Float32, Float16 and Q8_0 GGML tensors (got {result.ElementType}).");

            float* buffer = (float*)GetBufferStart(result);
            TensorIterState iter = new TensorIterState(buffer, result.DimensionCount, result.SizesMemory, result.StridesMemory);

            do
            {
                for (; !iter.ReachedBlockEnd(); iter.BlockStep())
                {
                    *iter.data = value;
                }
            } while (iter.NextBlock());
        }

        private static bool IsContiguousNonNarrowed(Tensor t)
        {
            if (t.StorageOffset != 0) return false;
            long expected = 1;
            for (int d = t.DimensionCount - 1; d >= 0; d--)
            {
                if (t.Strides[d] != expected) return false;
                expected *= t.Sizes[d];
            }
            return expected == t.ElementCount();
        }

        [RegisterOpStorageType("buildtrimask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildTriMask(Tensor result, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildtrimask");
            GetFlatRowsCols(result, "buildtrimask", out int rows, out int cols);

            float* resultPtr = (float*)GetBufferStart(result);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = col <= row ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildselfmask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSelfMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildselfmask");
            ValidateMaskLengthsTensor(originalLengths, nameof(originalLengths), "buildselfmask");
            GetFlatRowsCols(result, "buildselfmask", out int rows, out int cols);

            if (paddedSeqLen <= 0 || (rows % paddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildselfmask expects rows to be divisible by paddedSeqLen.");
            }

            int batchSize = rows / paddedSeqLen;
            if (originalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildselfmask expects one original length per batch item.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* originalLengthsPtr = (float*)GetBufferStart(originalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / paddedSeqLen;
                int seqIdxInBatch = row % paddedSeqLen;
                int originalLength = (int)originalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < originalLength && seqIdxInBatch < originalLength) ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildselftrimask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSelfTriMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildselftrimask");
            ValidateMaskLengthsTensor(originalLengths, nameof(originalLengths), "buildselftrimask");
            GetFlatRowsCols(result, "buildselftrimask", out int rows, out int cols);

            if (paddedSeqLen <= 0 || (rows % paddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildselftrimask expects rows to be divisible by paddedSeqLen.");
            }

            int batchSize = rows / paddedSeqLen;
            if (originalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildselftrimask expects one original length per batch item.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* originalLengthsPtr = (float*)GetBufferStart(originalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / paddedSeqLen;
                int seqIdxInBatch = row % paddedSeqLen;
                int originalLength = (int)originalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < originalLength && seqIdxInBatch < originalLength && col <= seqIdxInBatch) ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildsrctgtmask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLen, int tgtPaddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildsrctgtmask");
            ValidateMaskLengthsTensor(srcOriginalLengths, nameof(srcOriginalLengths), "buildsrctgtmask");
            ValidateMaskLengthsTensor(tgtOriginalLengths, nameof(tgtOriginalLengths), "buildsrctgtmask");
            GetFlatRowsCols(result, "buildsrctgtmask", out int rows, out int cols);

            if (tgtPaddedSeqLen <= 0 || (rows % tgtPaddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildsrctgtmask expects rows to be divisible by tgtPaddedSeqLen.");
            }

            int batchSize = rows / tgtPaddedSeqLen;
            if (srcOriginalLengths.ElementCount() != batchSize || tgtOriginalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildsrctgtmask expects source and target length tensors to match batch size.");
            }

            if (cols != srcPaddedSeqLen)
            {
                throw new InvalidOperationException("buildsrctgtmask expects the result last dimension to equal srcPaddedSeqLen.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* srcOriginalLengthsPtr = (float*)GetBufferStart(srcOriginalLengths);
            float* tgtOriginalLengthsPtr = (float*)GetBufferStart(tgtOriginalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / tgtPaddedSeqLen;
                int seqIdxInBatch = row % tgtPaddedSeqLen;
                int srcOriginalLength = (int)srcOriginalLengthsPtr[batchIdx];
                int tgtOriginalLength = (int)tgtOriginalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < srcOriginalLength && seqIdxInBatch < tgtOriginalLength) ? value : maskedValue;
                }
            }

            return result;
        }

        // Detects the common 2D contiguous float32 layout that lets us
        // bypass TensorDimIterState. result/src/indices must all be 2D
        // contiguous and indices/result must share shape (the public API
        // already guarantees this). The fast paths walk rows in the outer
        // loop (cache-friendly writes) and, for dim==0, fall back to a
        // single row memcpy when all indices in a row are equal (the
        // embedding-lookup pattern).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryGather2DFastSetup(
            Tensor result, Tensor src, Tensor indices, int dim,
            out float* rPtr, out float* sPtr, out byte* iPtr, out bool indicesInt32,
            out long N, out long D, out long otherDim)
        {
            rPtr = sPtr = null;
            iPtr = null;
            indicesInt32 = false;
            N = D = otherDim = 0;

            if (result == null || src == null || indices == null) return false;
            if (result.DimensionCount != 2 || src.DimensionCount != 2 || indices.DimensionCount != 2) return false;
            if (dim != 0 && dim != 1) return false;
            if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32) return false;

            if (indices.ElementType == DType.Int32) indicesInt32 = true;
            else if (indices.ElementType != DType.Float32) return false;

            if (!IsContiguousNonNarrowed(result) || !IsContiguousNonNarrowed(src) || !IsContiguousNonNarrowed(indices))
                return false;

            if (result.Sizes[0] != indices.Sizes[0] || result.Sizes[1] != indices.Sizes[1]) return false;
            int otherIdx = dim == 0 ? 1 : 0;
            if (src.Sizes[otherIdx] != result.Sizes[otherIdx]) return false;

            N = result.Sizes[0];
            D = result.Sizes[1];
            otherDim = src.Sizes[dim];

            rPtr = (float*)GetBufferStart(result);
            sPtr = (float*)GetBufferStart(src);
            iPtr = (byte*)GetBufferStart(indices);
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ReadIndex(byte* iPtr, bool isInt32, long off)
        {
            return isInt32 ? ((int*)iPtr)[off] : (long)((float*)iPtr)[off];
        }

        [RegisterOpStorageType("gather", typeof(GgmlStorage))]
        public static unsafe Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateGatherArguments(result, src, dim, indices);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);

            if (TryGather2DFastSetup(writeTarget, src, indices, dim,
                out float* rPtr, out float* sPtr, out byte* iPtr, out bool isInt32,
                out long N, out long D, out long sSize))
            {
                if (dim == 0)
                {
                    // result[i, j] = src[indices[i, j], j]
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* resRow = rPtr + rowOff;
                        long firstIdx = ReadIndex(iPtr, isInt32, rowOff);
                        if (firstIdx < 0 || firstIdx >= sSize)
                            throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{firstIdx}', srcSize = '{sSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if (ReadIndex(iPtr, isInt32, rowOff + j) != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            long bytes = D * sizeof(float);
                            Buffer.MemoryCopy(sPtr + firstIdx * D, resRow, bytes, bytes);
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                                if (idx < 0 || idx >= sSize)
                                    throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', srcSize = '{sSize}'");
                                resRow[j] = sPtr[idx * D + j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, j] = src[i, indices[i, j]]
                {
                    long srcCols = sSize;
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* srcRow = sPtr + i * srcCols;
                        float* resRow = rPtr + rowOff;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                            if (idx < 0 || idx >= srcCols)
                                throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', srcSize = '{srcCols}'");
                            resRow[j] = srcRow[idx];
                        }
                    }
                }
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.SizesMemory, src.StridesMemory, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.SizesMemory, indices.StridesMemory, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = ReadIndexValue(indicesIter, indices.ElementType, i);
                    if (idx < 0 || idx >= srcIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', srcSize = '{srcIter.size}'");
                    }

                    *(resultIter.data + i * resultIter.stride) = *(srcIter.data + idx * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("scatter", typeof(GgmlStorage))]
        public static unsafe Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateScatterArguments(result, src, dim, indices, "scatter");

            if (TryGather2DFastSetup(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out byte* iPtr, out bool isInt32,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] = src[i, j]
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* srcRow = sPtr + rowOff;
                        long firstIdx = ReadIndex(iPtr, isInt32, rowOff);
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{firstIdx}', resultSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if (ReadIndex(iPtr, isInt32, rowOff + j) != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            long bytes = D * sizeof(float);
                            Buffer.MemoryCopy(srcRow, rPtr + firstIdx * D, bytes, bytes);
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', resultSize = '{rSize}'");
                                rPtr[idx * D + j] = srcRow[j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, indices[i, j]] = src[i, j]
                {
                    long resCols = rSize;
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* resRow = rPtr + i * resCols;
                        float* srcRow = sPtr + rowOff;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', resultSize = '{resCols}'");
                            resRow[idx] = srcRow[j];
                        }
                    }
                }
                return result;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.SizesMemory, result.StridesMemory, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.SizesMemory, src.StridesMemory, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.SizesMemory, indices.StridesMemory, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = ReadIndexValue(indicesIter, indices.ElementType, i);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) = *(srcIter.data + i * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("scatter_add", typeof(GgmlStorage))]
        public static unsafe Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateScatterArguments(result, src, dim, indices, "scatter_add");

            if (TryGather2DFastSetup(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out byte* iPtr, out bool isInt32,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] += src[i, j]
                    int vectorSize = System.Numerics.Vector<float>.Count;
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* srcRow = sPtr + rowOff;
                        long firstIdx = ReadIndex(iPtr, isInt32, rowOff);
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter_add. Idx = '{firstIdx}', resultSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if (ReadIndex(iPtr, isInt32, rowOff + j) != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            float* dst = rPtr + firstIdx * D;
                            long j = 0;
                            for (; j <= D - vectorSize; j += vectorSize)
                            {
                                var a = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(ref *(byte*)(dst + j));
                                var b = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(ref *(byte*)(srcRow + j));
                                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref *(byte*)(dst + j), a + b);
                            }
                            for (; j < D; j++)
                                dst[j] += srcRow[j];
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter_add. Idx = '{idx}', resultSize = '{rSize}'");
                                rPtr[idx * D + j] += srcRow[j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, indices[i, j]] += src[i, j]
                {
                    long resCols = rSize;
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* resRow = rPtr + i * resCols;
                        float* srcRow = sPtr + rowOff;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = ReadIndex(iPtr, isInt32, rowOff + j);
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter_add. Idx = '{idx}', resultSize = '{resCols}'");
                            resRow[idx] += srcRow[j];
                        }
                    }
                }
                return result;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.SizesMemory, result.StridesMemory, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.SizesMemory, src.StridesMemory, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.SizesMemory, indices.StridesMemory, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = ReadIndexValue(indicesIter, indices.ElementType, i);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter_add. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) += *(srcIter.data + i * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("scatter_fill", typeof(GgmlStorage))]
        public static unsafe Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            ValidateScatterFillArguments(result, dim, indices);

            // 2D-contig fast path (no src tensor in the signature).
            if (result.DimensionCount == 2 && indices.DimensionCount == 2
                && (dim == 0 || dim == 1)
                && result.ElementType == DType.Float32
                && (indices.ElementType == DType.Int32 || indices.ElementType == DType.Float32)
                && IsContiguousNonNarrowed(result) && IsContiguousNonNarrowed(indices)
                && indices.Sizes[1 - dim] == result.Sizes[1 - dim])
            {
                bool isInt32 = indices.ElementType == DType.Int32;
                long N = indices.Sizes[0];
                long D = indices.Sizes[1];
                long otherDim = result.Sizes[dim];

                float* rPtrFast = (float*)GetBufferStart(result);
                byte* iPtrFast = (byte*)GetBufferStart(indices);

                if (dim == 0)
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = ReadIndex(iPtrFast, isInt32, rowOff + j);
                            if (idx < 0 || idx >= otherDim)
                                throw new IndexOutOfRangeException($"Invalid index in scatter_fill. Idx = '{idx}', resultSize = '{otherDim}'");
                            rPtrFast[idx * resCols + j] = value;
                        }
                    }
                }
                else // dim == 1
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        long rowOff = i * D;
                        float* resRow = rPtrFast + i * resCols;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = ReadIndex(iPtrFast, isInt32, rowOff + j);
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter_fill. Idx = '{idx}', resultSize = '{resCols}'");
                            resRow[idx] = value;
                        }
                    }
                }
                return result;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.SizesMemory, result.StridesMemory, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.SizesMemory, indices.StridesMemory, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = ReadIndexValue(indicesIter, indices.ElementType, i);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter_fill. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) = value;
                }
            } while (resultIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("addmm", typeof(GgmlStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            ValidateAddmmArguments(result, src, m1, m2);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView2D srcView = default;
            Tensor compactSrc = null;
            try
            {
                if (!TryCreateStandardView(writeTarget, out GgmlTensorView2D resultView)
                    || (beta != 0.0f && !TryCreateStandardOrBroadcastView(src, out srcView, out compactSrc))
                    || !TryCreateRawView(m1, out GgmlTensorView2D m1View)
                    || !TryCreateRawView(m2, out GgmlTensorView2D m2View))
                {
                    throw new NotSupportedException("GGML addmm requires Float32 tensors with supported row-contiguous/view-compatible layouts.");
                }

                if (beta == 0.0f)
                {
                    srcView = default;
                }

                GgmlNative.Addmm(resultView, srcView, m1View, m2View, beta, alpha);
            }
            finally
            {
                compactSrc?.Dispose();
            }

            return writeTarget;
        }

        /// <summary>
        /// Quantized linear forward: result = m1 * quantizedWeight^T.
        /// The quantized weight data is stored in native GGUF format with GGML type.
        /// ne0 = inDim (shared dimension), ne1 = outDim.
        /// </summary>
        public static void AddmmQuant(Tensor result, Tensor m1, IntPtr weightData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (result.DimensionCount != 2 || m1.DimensionCount != 2)
                throw new ArgumentException("AddmmQuant requires 2D tensors for result and m1.");
            if (m1.ElementType != DType.Float32 || result.ElementType != DType.Float32)
                throw new ArgumentException("AddmmQuant requires Float32 tensors for result and m1.");
            if (!HasNativeBufferStorage(m1) || !HasNativeBufferStorage(result))
                throw new ArgumentException("AddmmQuant requires tensors backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateRawView(m1, out GgmlTensorView2D m1View))
            {
                throw new NotSupportedException("GGML AddmmQuant requires tensors with supported row-contiguous layouts.");
            }

            GgmlNative.AddmmQuant(resultView, m1View, weightData, ggmlType, ne0, ne1, rawBytes);
        }

        /// <summary>
        /// Fused RMSNorm + quantized MatMul in a single GPU dispatch.
        /// result = matmul(rms_norm(input, normWeight, eps), quantWeight)
        /// </summary>
        public static void FusedRmsNormMatMulQuant(Tensor result, Tensor input, Tensor normWeight, float eps,
            IntPtr weightData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (result.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("FusedRmsNormMatMulQuant requires 2D tensors.");
            if (input.ElementType != DType.Float32 || result.ElementType != DType.Float32)
                throw new ArgumentException("FusedRmsNormMatMulQuant requires Float32 tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(result) || !HasNativeBufferStorage(normWeight))
                throw new ArgumentException("FusedRmsNormMatMulQuant requires native-backed tensors.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("FusedRmsNormMatMulQuant requires supported layouts.");

            int normCount = (int)normWeight.ElementCount();
            IntPtr normPtr = GetBufferStart(normWeight);

            GgmlNative.FusedRmsNormMatMulQuant(resultView, inputView, normPtr, normCount, eps,
                weightData, ggmlType, ne0, ne1, rawBytes);
        }

        /// <summary>
        /// Fused quantized MatMul + Add in a single GPU dispatch.
        /// residual += matmul(input, quantWeight)
        /// </summary>
        public static void FusedMatMulQuantAdd(Tensor residual, Tensor input,
            IntPtr weightData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (residual.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("FusedMatMulQuantAdd requires 2D tensors.");
            if (input.ElementType != DType.Float32 || residual.ElementType != DType.Float32)
                throw new ArgumentException("FusedMatMulQuantAdd requires Float32 tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(residual))
                throw new ArgumentException("FusedMatMulQuantAdd requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("FusedMatMulQuantAdd requires supported layouts.");

            GgmlNative.FusedMatMulQuantAdd(residualView, inputView,
                weightData, ggmlType, ne0, ne1, rawBytes);
        }

        /// <summary>
        /// Fused RMSNorm + residual add in a single GGML graph dispatch:
        /// residual += rms_norm(input, normWeight, eps). <paramref name="residual"/> and
        /// <paramref name="input"/> are distinct 2D Float32 tensors. Collapses the
        /// Ops.RMSNorm + Ops.Add pair (Gemma's 3 post-norm-add sites per layer) into one
        /// dispatch on the GGML backend, mirroring the MLX / pure-CUDA fused paths.
        /// </summary>
        public static void RmsNormResidualAdd(Tensor residual, Tensor input, Tensor normWeight, float eps)
        {
            if (residual.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("RmsNormResidualAdd requires 2D tensors.");
            if (input.ElementType != DType.Float32 || residual.ElementType != DType.Float32 || normWeight.ElementType != DType.Float32)
                throw new ArgumentException("RmsNormResidualAdd requires Float32 tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(residual) || !HasNativeBufferStorage(normWeight))
                throw new ArgumentException("RmsNormResidualAdd requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("RmsNormResidualAdd requires supported layouts.");

            GgmlNative.FusedRmsNormResidualAdd(residualView, inputView,
                GetBufferStart(normWeight), (int)normWeight.ElementCount(), eps);
        }

        /// <summary>
        /// Fused Gemma Per-Layer-Embeddings block in one GGML graph:
        ///   residual += rms_norm(proj_W^T @ (gelu(inp_gate_W^T @ residual) * perLayerInput), postNorm).
        /// inp_gate is [hidden, ple_dim], proj is [ple_dim, hidden]; perLayerInput is
        /// [rows, ple_dim] F32. Collapses the PLE matmul/gelu/mul/proj/norm/add chain
        /// into one dispatch. residual is the inp_gate input AND the accumulator.
        /// </summary>
        public static void FusedPleBlockQuant(Tensor residual, Tensor perLayerInput,
            IntPtr inpGateData, int inpGateGgmlType, long inpGateNe0, long inpGateNe1, long inpGateRawBytes,
            IntPtr projData, int projGgmlType, long projNe0, long projNe1, long projRawBytes,
            Tensor postNorm, float eps)
        {
            if (residual.DimensionCount != 2 || perLayerInput.DimensionCount != 2)
                throw new ArgumentException("FusedPleBlockQuant requires 2D residual/perLayerInput tensors.");
            if (residual.ElementType != DType.Float32 || perLayerInput.ElementType != DType.Float32 || postNorm.ElementType != DType.Float32)
                throw new ArgumentException("FusedPleBlockQuant requires Float32 residual/perLayerInput/postNorm.");
            if (!HasNativeBufferStorage(residual) || !HasNativeBufferStorage(perLayerInput) || !HasNativeBufferStorage(postNorm))
                throw new ArgumentException("FusedPleBlockQuant requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(perLayerInput, out GgmlTensorView2D pliView))
                throw new NotSupportedException("FusedPleBlockQuant requires supported tensor layouts.");

            GgmlNative.FusedPleBlockQuant(residualView, pliView,
                inpGateData, inpGateGgmlType, inpGateNe0, inpGateNe1, inpGateRawBytes,
                projData, projGgmlType, projNe0, projNe1, projRawBytes,
                GetBufferStart(postNorm), (int)postNorm.ElementCount(), eps);
        }

        /// <summary>
        /// Fully fused dense SwiGLU FFN with residual add in a single GGML graph dispatch.
        /// residual += down_W^T @ ( silu(gate_part) * up_part ),
        ///   where [gate_part | up_part] = gate_up_W^T @ rms_norm(input, normW, eps).
        /// Replaces the previous chain of FusedRmsNormMatMulQuant + SiLUMul(Split) +
        /// FusedMatMulQuantAdd (3 dispatches) with one, removing the matching number of
        /// graph builds, allocations and synchronization round trips.
        /// </summary>
        public static void FusedFFNSwiGLUQuant(Tensor residual, Tensor input,
            Tensor normWeight, float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim)
        {
            if (residual.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("FusedFFNSwiGLUQuant requires 2D residual/input tensors.");
            if (input.ElementType != DType.Float32 || residual.ElementType != DType.Float32 || normWeight.ElementType != DType.Float32)
                throw new ArgumentException("FusedFFNSwiGLUQuant requires Float32 tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(residual) || !HasNativeBufferStorage(normWeight))
                throw new ArgumentException("FusedFFNSwiGLUQuant requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("FusedFFNSwiGLUQuant requires supported tensor layouts.");

            int normCount = (int)normWeight.ElementCount();
            IntPtr normPtr = GetBufferStart(normWeight);

            GgmlNative.FusedFFNSwiGLUQuant(residualView, inputView, normPtr, normCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim);
        }

        /// <summary>
        /// Fused dense FFN projection (no residual fold) in a single GGML graph dispatch:
        ///   output = down_W^T @ ( act(gate_part) * up_part ),
        ///   where [gate_part | up_part] = gate_up_W^T @ rms_norm(input, normW, eps).
        /// <paramref name="actType"/>: 0 = SiLU (SwiGLU), 1 = GELU tanh (GeGLU, Gemma 4).
        /// Unlike <see cref="FusedFFNSwiGLUQuant"/> this writes the projection to a
        /// separate <paramref name="output"/> tensor so callers can apply a post-FFN norm
        /// before the residual add (Gemma 4). Keeps the large gate_up intermediate on-device.
        /// </summary>
        public static void FusedFFNActProjectQuant(Tensor output, Tensor input,
            Tensor normWeight, float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim, int actType)
        {
            if (output.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("FusedFFNActProjectQuant requires 2D output/input tensors.");
            if (input.ElementType != DType.Float32 || output.ElementType != DType.Float32 || normWeight.ElementType != DType.Float32)
                throw new ArgumentException("FusedFFNActProjectQuant requires Float32 tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(output) || !HasNativeBufferStorage(normWeight))
                throw new ArgumentException("FusedFFNActProjectQuant requires native-backed tensors.");

            if (!TryCreateStandardView(output, out GgmlTensorView2D outputView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("FusedFFNActProjectQuant requires supported tensor layouts.");

            int normCount = (int)normWeight.ElementCount();
            IntPtr normPtr = GetBufferStart(normWeight);

            GgmlNative.FusedFFNActProjectQuant(outputView, inputView, normPtr, normCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim, actType);
        }

        /// <summary>
        /// Fused output projection + residual + RMSNorm + router projection in one dispatch.
        /// For MoE decode: replaces TryLinearAddInto + RMSNorm + LinearForwardCached (3 ops -> 1).
        /// </summary>
        public static void FusedOutProjNormRouter(
            Tensor residual, Tensor input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            Tensor normWeight, float eps,
            Tensor normedOutput,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerRawBytes,
            Tensor routerOutput)
        {
            if (!HasNativeBufferStorage(residual) || !HasNativeBufferStorage(input)
                || !HasNativeBufferStorage(normWeight) || !HasNativeBufferStorage(normedOutput)
                || !HasNativeBufferStorage(routerOutput))
                throw new ArgumentException("FusedOutProjNormRouter requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView)
                || !TryCreateStandardView(normedOutput, out GgmlTensorView2D normedView)
                || !TryCreateStandardView(routerOutput, out GgmlTensorView2D routerView))
                throw new NotSupportedException("FusedOutProjNormRouter requires supported layouts.");

            int normCount = (int)normWeight.ElementCount();
            IntPtr normPtr = GetBufferStart(normWeight);

            GgmlNative.FusedOutProjNormRouter(residualView, inputView,
                outProjData, outProjType, outNe0, outNe1, outRawBytes,
                normPtr, normCount, eps, normedView,
                routerData, routerType, routerNe0, routerNe1, routerRawBytes,
                routerView);
        }

        /// <summary>
        /// Fused output projection + residual + RMSNorm + FFN SwiGLU + residual in one dispatch.
        /// Replaces TryLinearAddInto + FusedFFNSwiGLUQuant (2 dispatches -> 1).
        /// </summary>
        public static void FusedOutProjFFN(
            Tensor residual, Tensor input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            Tensor ffnNormWeight, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim)
        {
            if (!HasNativeBufferStorage(residual) || !HasNativeBufferStorage(input) || !HasNativeBufferStorage(ffnNormWeight))
                throw new ArgumentException("FusedOutProjFFN requires native-backed tensors.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
                throw new NotSupportedException("FusedOutProjFFN requires supported layouts.");

            int normCount = (int)ffnNormWeight.ElementCount();
            IntPtr normPtr = GetBufferStart(ffnNormWeight);

            GgmlNative.FusedOutProjFFN(residualView, inputView,
                outProjData, outProjType, outNe0, outNe1, outRawBytes,
                normPtr, normCount, eps,
                guData, guType, guNe0, guNe1, guRawBytes,
                dnData, dnType, dnNe0, dnNe1, dnRawBytes,
                halfDim);
        }

        /// <summary>
        /// Fused vision encoder MLP: LayerNorm + up + bias + GELU + down + bias + residual
        /// as a single GGML graph dispatch. Replaces 7 separate dispatches with 1.
        /// </summary>
        public static unsafe void FusedVisionMLP(
            Tensor hidden, Tensor lnWeight, Tensor lnBias, float eps,
            Tensor upWeight, Tensor upBias,
            Tensor downWeight, Tensor downBias)
        {
            if (!HasNativeBufferStorage(hidden))
                throw new ArgumentException("FusedVisionMLP requires native-backed hidden tensor.");

            if (!TryCreateStandardView(hidden, out GgmlTensorView2D hiddenView))
                throw new NotSupportedException("FusedVisionMLP requires standard layout for hidden.");

            IntPtr lnWPtr = GetBufferStart(lnWeight);
            IntPtr lnBPtr = GetBufferStart(lnBias);
            int lnDim = (int)lnWeight.ElementCount();

            IntPtr upWPtr = GetBufferStart(upWeight);
            int upNe0 = (int)upWeight.Sizes[upWeight.DimensionCount - 1];
            int upNe1 = (int)upWeight.Sizes[0];
            long upBytes = upWeight.ElementCount() * sizeof(float);
            IntPtr upBPtr = GetBufferStart(upBias);
            int upBDim = (int)upBias.ElementCount();

            IntPtr downWPtr = GetBufferStart(downWeight);
            int downNe0 = (int)downWeight.Sizes[downWeight.DimensionCount - 1];
            int downNe1 = (int)downWeight.Sizes[0];
            long downBytes = downWeight.ElementCount() * sizeof(float);
            IntPtr downBPtr = GetBufferStart(downBias);
            int downBDim = (int)downBias.ElementCount();

            GgmlNative.FusedVisionMLP(hiddenView,
                lnWPtr, lnBPtr, lnDim, eps,
                upWPtr, upNe0, upNe1, upBytes, upBPtr, upBDim,
                downWPtr, downNe0, downNe1, downBytes, downBPtr, downBDim);
        }

        /// <summary>
        /// Fused vision attention: LN + QKV + bias + RoPE + SDPA + out + bias + residual
        /// as a single GGML graph dispatch. Replaces ~8 separate dispatches with 1.
        /// </summary>
        public static unsafe void FusedVisionAttention(
            Tensor hidden, Tensor lnWeight, Tensor lnBias, float eps,
            Tensor qkvWeight, Tensor qkvBias,
            Tensor outWeight, Tensor outBias,
            float[] cosTable, float[] sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale)
        {
            if (!HasNativeBufferStorage(hidden))
                throw new ArgumentException("FusedVisionAttention requires native-backed hidden tensor.");

            if (!TryCreateStandardView(hidden, out GgmlTensorView2D hiddenView))
                throw new NotSupportedException("FusedVisionAttention requires standard layout for hidden.");

            IntPtr lnWPtr = GetBufferStart(lnWeight);
            IntPtr lnBPtr = GetBufferStart(lnBias);
            int lnDim = (int)lnWeight.ElementCount();

            IntPtr qkvWPtr = GetBufferStart(qkvWeight);
            int qkvNe0 = (int)qkvWeight.Sizes[qkvWeight.DimensionCount - 1];
            int qkvNe1 = (int)qkvWeight.Sizes[0];
            long qkvBytes = qkvWeight.ElementCount() * sizeof(float);
            IntPtr qkvBPtr = GetBufferStart(qkvBias);
            int qkvBDim = (int)qkvBias.ElementCount();

            IntPtr outWPtr = GetBufferStart(outWeight);
            int outNe0 = (int)outWeight.Sizes[outWeight.DimensionCount - 1];
            int outNe1 = (int)outWeight.Sizes[0];
            long outBytes = outWeight.ElementCount() * sizeof(float);
            IntPtr outBPtr = GetBufferStart(outBias);
            int outBDim = (int)outBias.ElementCount();

            fixed (float* cosPtr = cosTable, sinPtr = sinTable)
            {
                GgmlNative.FusedVisionAttention(hiddenView,
                    lnWPtr, lnBPtr, lnDim, eps,
                    qkvWPtr, qkvNe0, qkvNe1, qkvBytes, qkvBPtr, qkvBDim,
                    outWPtr, outNe0, outNe1, outBytes, outBPtr, outBDim,
                    (IntPtr)cosPtr, (IntPtr)sinPtr,
                    numPatches, numHeads, headDim, halfDim, attnScale);
            }
        }

        /// <summary>
        /// Native row selection (embedding lookup) from a quantized tensor.
        /// Uses GGML's ggml_get_rows which dequantizes on-the-fly (GPU-accelerated on Metal).
        /// </summary>
        public static void GetRowsQuant(Tensor result, IntPtr weightData, int ggmlType, long ne0, long ne1, long rawBytes, Tensor indices)
        {
            if (result.DimensionCount != 2 || result.ElementType != DType.Float32)
                throw new ArgumentException("GetRowsQuant requires a 2D Float32 result tensor.");
            if (!HasNativeBufferStorage(result))
                throw new ArgumentException("GetRowsQuant requires a result tensor backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateContiguousTensor(indices, out GgmlContiguousTensor indexTensor, DType.Int32))
            {
                throw new NotSupportedException("GGML GetRowsQuant requires a standard-layout result and contiguous Int32 indices.");
            }

            GgmlNative.GetRowsQuant(resultView, weightData, ggmlType, ne0, ne1, rawBytes, indexTensor);
        }

        /// <summary>
        /// Batched MoE expert forward: processes all selected experts in a single GGML graph.
        /// For each expert: up_proj -> relu_squared -> down_proj -> scale(route_weight) -> accumulate.
        /// Reduces N*2 GPU dispatches to 1 per MoE layer.
        /// </summary>
        public static void MoEExpertsForward(Tensor result, Tensor input,
            int numExperts, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            if (result.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("MoEExpertsForward requires 2D tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(result))
                throw new ArgumentException("MoEExpertsForward requires tensors backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
            {
                throw new NotSupportedException("MoEExpertsForward requires tensors with supported row-contiguous layouts.");
            }

            GgmlNative.MoEExpertsForward(resultView, inputView, numExperts,
                upDataPtrs, downDataPtrs,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights);
        }

        /// <summary>
        /// Batched MoE expert forward with SwiGLU activation (Qwen3 / Mixtral style).
        /// For each expert: gate_proj -> silu(gate) * up_proj(input) -> down_proj -> scale(route_weight) -> accumulate.
        /// Reduces 4*N GPU dispatches to 1 per MoE layer.
        /// </summary>
        public static void MoEExpertsSwiGLUForward(Tensor result, Tensor input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            if (result.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("MoEExpertsSwiGLUForward requires 2D tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(result))
                throw new ArgumentException("MoEExpertsSwiGLUForward requires tensors backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
            {
                throw new NotSupportedException("MoEExpertsSwiGLUForward requires tensors with supported row-contiguous layouts.");
            }

            GgmlNative.MoEExpertsSwiGLUForward(resultView, inputView, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights);
        }

        /// <summary>
        /// Extended SwiGLU MoE forward that fuses (a) routed expert computation, (b) optional
        /// shared expert computation, and (c) residual accumulation into one GGML graph dispatch.
        /// Saves up to 4*N + 4 + 1 dispatches per MoE layer (versus the per-expert path).
        /// </summary>
        public static void MoEExpertsSwiGLUResidual(Tensor residual, Tensor input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights,
            bool useShared,
            IntPtr sharedGateData, IntPtr sharedUpData, IntPtr sharedDownData,
            int sharedGateGgmlType, long sharedGateNe0, long sharedGateNe1, long sharedGateRawBytes,
            int sharedUpGgmlType, long sharedUpNe0, long sharedUpNe1, long sharedUpRawBytes,
            int sharedDownGgmlType, long sharedDownNe0, long sharedDownNe1, long sharedDownRawBytes,
            float sharedScalar)
        {
            if (residual.DimensionCount != 2 || input.DimensionCount != 2)
                throw new ArgumentException("MoEExpertsSwiGLUResidual requires 2D tensors.");
            if (!HasNativeBufferStorage(input) || !HasNativeBufferStorage(residual))
                throw new ArgumentException("MoEExpertsSwiGLUResidual requires tensors backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(residual, out GgmlTensorView2D residualView)
                || !TryCreateRawView(input, out GgmlTensorView2D inputView))
            {
                throw new NotSupportedException("MoEExpertsSwiGLUResidual requires tensors with supported row-contiguous layouts.");
            }

            GgmlNative.MoEExpertsSwiGLUResidual(residualView, inputView, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights,
                useShared,
                sharedGateData, sharedUpData, sharedDownData,
                sharedGateGgmlType, sharedGateNe0, sharedGateNe1, sharedGateRawBytes,
                sharedUpGgmlType, sharedUpNe0, sharedUpNe1, sharedUpRawBytes,
                sharedDownGgmlType, sharedDownNe0, sharedDownNe1, sharedDownRawBytes,
                sharedScalar);
        }

        /// <summary>
        /// Batched quantized matmul: processes multiple sub-weights at different offsets within a single quantized blob.
        /// </summary>
        public static void AddmmQuantBatch(Tensor result, Tensor m1, IntPtr weightData, int ggmlType, long ne0, long rawBytes,
            int batchCount, long[] weightOffsets, long[] weightNe1Arr)
        {
            if (result.DimensionCount != 2 || m1.DimensionCount != 2)
                throw new ArgumentException("AddmmQuantBatch requires 2D tensors.");
            if (!HasNativeBufferStorage(m1) || !HasNativeBufferStorage(result))
                throw new ArgumentException("AddmmQuantBatch requires tensors backed by native CPU-accessible storage.");

            if (!TryCreateStandardView(result, out GgmlTensorView2D resultView)
                || !TryCreateRawView(m1, out GgmlTensorView2D m1View))
            {
                throw new NotSupportedException("GGML AddmmQuantBatch requires tensors with supported row-contiguous layouts.");
            }

            GgmlNative.AddmmQuantBatch(resultView, m1View, weightData, ggmlType, ne0, rawBytes, batchCount, weightOffsets, weightNe1Arr);
        }

        public static void PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            GgmlNative.PreloadQuantizedWeight(cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
        }

        public static void RegisterOffloadable(IntPtr key) => GgmlNative.RegisterOffloadable(key);
        public static void SetOffloadableBudget(long bytes) => GgmlNative.SetOffloadableBudget(bytes);
        public static void ClearOffloadableState() => GgmlNative.ClearOffloadableState();
        public static void SetDeviceCopyBudget(long bytes) => GgmlNative.SetDeviceCopyBudget(bytes);
        public static bool TryGetDeviceMemoryInfo(out long freeBytes, out long totalBytes)
            => GgmlNative.TryGetDeviceMemoryInfo(out freeBytes, out totalBytes);
        public static bool TryRegisterPinnedHostBuffer(IntPtr ptr, long bytes)
            => GgmlNative.TryRegisterPinnedHostBuffer(ptr, bytes);
        public static void UnregisterPinnedHostBuffer(IntPtr ptr) => GgmlNative.UnregisterPinnedHostBuffer(ptr);

        public static IntPtr AlignedAlloc(long size) => GgmlNative.AlignedAlloc(size);
        public static void AlignedFree(IntPtr ptr) => GgmlNative.AlignedFree(ptr);
        public static void ClearHostBufferCache() => GgmlNative.ClearHostBufferCache();
        public static void Shutdown() => GgmlNative.Shutdown();
        public static void InvalidateHostBuffer(IntPtr ptr) => GgmlNative.InvalidateHostBuffer(ptr);
        public static long DeviceCopyCacheResidentBytes() => GgmlNative.DeviceCopyCacheResidentBytes();
        public static bool TryGetBackendMemory(out long freeBytes, out long totalBytes) => GgmlNative.TryGetBackendMemory(out freeBytes, out totalBytes);
        public static void SyncHostBuffer(IntPtr ptr, long byteCount) => GgmlNative.SyncHostBuffer(ptr, byteCount);
        public static void SetAsyncCompute(bool enabled) => GgmlNative.SetAsyncCompute(enabled);
        public static bool GetAsyncCompute() => GgmlNative.GetAsyncCompute();
        public static void HostReadBarrier() => GgmlNative.HostReadBarrier();
        public static bool CanInitializeBackend(GgmlBackendType backendType) => GgmlNative.CanInitialize(backendType);
        public static void EnsureBackendAvailable(GgmlBackendType backendType) => GgmlNative.EnsureAvailable(backendType);

        public static void TransformerModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            int oType, long oNe0, long oNe1, long oBytes,
            int guType, long guNe0, long guNe1, long guBytes,
            int downType, long downNe0, long downNe1, long downBytes,
            int headDim, int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            GgmlNative.TransformerModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, ffnNormArr, guArr, downArr,
                kCacheArr, vCacheArr,
                qkvType, qkvNe0, qkvNe1, qkvBytes,
                oType, oNe0, oNe1, oBytes,
                guType, guNe0, guNe1, guBytes,
                downType, downNe0, downNe1, downBytes,
                headDim, numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType);
        }

        /// <summary>
        /// Full transformer layer decode (seqLen=1) in a single GGML graph.
        /// Updates hidden state in-place and writes new K/V to the KV cache.
        /// </summary>
        public static void TransformerLayerDecode(
            IntPtr hiddenData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr ffnNormData,
            IntPtr guData, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            GgmlNative.TransformerLayerDecode(
                hiddenData, hiddenSize,
                attnNormData,
                qkvData, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormData, kNormData, headDim,
                oData, oType, oNe0, oNe1, oBytes,
                ffnNormData,
                guData, guType, guNe0, guNe1, guBytes,
                downData, downType, downNe0, downNe1, downBytes,
                kCacheData, vCacheData,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType);
        }

        /// <summary>
        /// Single-token flash attention decode kernel. Reads Q/K/V (post-norm, post-RoPE) for
        /// the new query position, appends K/V to the persistent KV cache at <paramref name="position"/>,
        /// and runs ggml_flash_attn_ext on the device against the populated portion of the cache.
        ///
        /// All input/output tensors must be F32 contiguous. Q, K, V, and the output are laid out
        /// as (heads, head_dim) in row-major order (i.e. head-contiguous). The KV caches use the
        /// usual layout (kv_heads, max_seq_len, head_dim).
        /// </summary>
        public static void FlashAttnDecode(Tensor q, Tensor k, Tensor v,
            Tensor kCache, Tensor vCache, Tensor output,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position, float scale)
        {
            if (q == null || k == null || v == null || kCache == null || vCache == null || output == null)
                throw new ArgumentNullException("Flash attention decode requires non-null Q/K/V/cache/output tensors.");
            if (q.ElementType != DType.Float32 || k.ElementType != DType.Float32 || v.ElementType != DType.Float32 ||
                output.ElementType != DType.Float32)
                throw new ArgumentException("Flash attention decode requires F32 Q/K/V/output tensors.");
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                throw new ArgumentException("Flash attention decode requires F32 or F16 K cache tensor.");
            if (vCache.ElementType != kCache.ElementType)
                throw new ArgumentException("Flash attention decode requires K and V caches to share an element type.");

            IntPtr qPtr = GetBufferStart(q);
            IntPtr kPtr = GetBufferStart(k);
            IntPtr vPtr = GetBufferStart(v);
            IntPtr kcPtr = GetBufferStart(kCache);
            IntPtr vcPtr = GetBufferStart(vCache);
            IntPtr outPtr = GetBufferStart(output);

            int kvGgmlType = kCache.ElementType == DType.Float16 ? 1 : 0;

            GgmlNative.FlashAttnDecode(
                qPtr, kPtr, vPtr,
                kcPtr, vcPtr, outPtr,
                numHeads, numKvHeads, headDim,
                maxSeqLen, position, scale, kvGgmlType);
        }

        /// <summary>
        /// Native batched paged-attention forward. Public wrapper around
        /// <see cref="GgmlNative.PagedAttentionForward(float[], float[], float[], float[], int[], int[], int[], int[], int[], int, int, int, int, int, int, float)"/>.
        /// Takes the paged K/V buffer as a flat row-major float[] (layout
        /// [num_blocks * block_size, num_kv_heads, head_dim]) and a per-
        /// sequence block table (concatenated flat ints + per-seq offsets).
        /// One Metal/CUDA kernel per sequence per layer.
        /// </summary>
        public static void PagedAttentionForward(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            GgmlNative.PagedAttentionForward(
                qData, pagedKData, pagedVData, outData,
                queryStartLoc, seqLens, positions,
                blockTableFlat, blockTableOffsets,
                numSeqs, numTokens, numHeads, numKvHeads, headDim,
                blockSize, scale, slidingWindow);
        }

        /// <summary>Native paged-attention forward with per-head attention
        /// sinks (gpt-oss style). Pass null for <paramref name="sinksData"/>
        /// to degenerate to the regular paged attention.</summary>
        public static void PagedAttentionForwardWithSinks(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            GgmlNative.PagedAttentionForwardWithSinks(
                qData, pagedKData, pagedVData, outData,
                queryStartLoc, seqLens, positions,
                blockTableFlat, blockTableOffsets,
                numSeqs, numTokens, numHeads, numKvHeads, headDim,
                blockSize, scale, slidingWindow, sinksData);
        }

        /// <summary>GPU-resident paged-attention forward. Q and OUT are
        /// passed as <see cref="IntPtr"/> into existing backend storage
        /// (typically <c>tensor.Storage.PtrAtElement(tensor.StorageOffset)</c>
        /// for a GGML-backed <see cref="Tensor"/>). The kernel zero-copy
        /// binds them to backend buffers so no host-side memcpy / sync
        /// happens for Q or OUT, eliminating the per-layer queue drain that
        /// <c>q.GetElementsAsFloat()</c> would otherwise force on the
        /// Gemma 4 batched path. K/V paged buffers stay as host arrays.
        /// </summary>
        public static void PagedAttentionForwardDevice(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            GgmlNative.PagedAttentionForwardDevice(
                qData, pagedKData, pagedVData, outData,
                queryStartLoc, seqLens, positions,
                blockTableFlat, blockTableOffsets,
                numSeqs, numTokens, numHeads, numKvHeads, headDim,
                blockSize, scale, slidingWindow);
        }

        /// <summary>GPU-resident paged-attention forward with per-head
        /// attention sinks. Pass <c>null</c> for <paramref name="sinksData"/>
        /// to match <see cref="PagedAttentionForwardDevice"/>.</summary>
        public static void PagedAttentionForwardDeviceWithSinks(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            GgmlNative.PagedAttentionForwardDeviceWithSinks(
                qData, pagedKData, pagedVData, outData,
                queryStartLoc, seqLens, positions,
                blockTableFlat, blockTableOffsets,
                numSeqs, numTokens, numHeads, numKvHeads, headDim,
                blockSize, scale, slidingWindow, sinksData);
        }

        /// <summary>
        /// Runs one entire Gemma 4 MoE transformer block (attention + dense
        /// shared FFN + in-graph-routed experts + all norms/residuals) as a
        /// single fused GGML graph on the device for decode (seqLen == 1).
        /// Replaces ~18-20 per-op dispatches per MoE layer, each of which would
        /// otherwise allocate/free a Metal buffer and synchronise. Throws on
        /// failure (caller falls back to the per-op TransformerBlock path).
        /// </summary>
        public static void Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs args)
        {
            GgmlNative.Gemma4MoELayerDecode(in args);
        }

        /// <summary>Fused DiffusionGemma decode layer (whole layer in one GGML graph). Returns false
        /// without throwing when the backend can't run it, so the caller falls back to the per-op path.</summary>
        public static bool TryDiffusionDecodeLayer(in DiffusionDecodeLayerArgs args)
        {
            return GgmlNative.TryDiffusionDecodeLayer(in args);
        }

        /// <summary>Fused DiffusionGemma lm_head tail (output_norm + lm_head + softcap) in one GGML graph.</summary>
        public static bool TryDiffusionLmHead(
            IntPtr hidden, int hiddenSize, int canvasLen,
            IntPtr outputNormW, IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float eps, float finalLogitSoftcap)
        {
            return GgmlNative.TryDiffusionLmHead(hidden, hiddenSize, canvasLen, outputNormW,
                lmHeadW, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, logitsOut, vocab, eps, finalLogitSoftcap);
        }

        /// <summary>Fused DiffusionGemma whole-model decode (all layers + lm_head in one graph). Returns
        /// false without throwing when the backend can't run it, so the caller falls back.</summary>
        public static bool TryDiffusionModelDecode(
            DiffusionDecodeLayerArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int canvasLen, int promptLen,
            IntPtr outputNormW, IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float finalLogitSoftcap)
        {
            return GgmlNative.TryDiffusionModelDecode(layers, numLayers, hidden, hiddenSize, canvasLen, promptLen,
                outputNormW, lmHeadW, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, logitsOut, vocab, finalLogitSoftcap);
        }

        /// <summary>
        /// Model-wide MoE decode: runs every layer (attention + dense FFN +
        /// in-graph MoE experts) as ONE GGML graph, dispatched/synchronised once
        /// per token instead of once per layer. Amortises the per-layer graph
        /// build / Metal encode / sync so the GPU stays saturated. Throws on
        /// failure (caller falls back to the per-layer path).
        /// </summary>
        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position)
        {
            GgmlNative.Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position);
        }

        /// <summary>
        /// Run the entire Gemma4 MoE transformer over <paramref name="numTokens"/>
        /// tokens (the MTP verify batch) as ONE GGML graph — the multi-token sibling
        /// of <see cref="Gemma4MoEModelDecode"/>. Reuses the same per-layer descriptor
        /// array; writes the per-row layer-stack hidden state (pre output_norm) back
        /// into <paramref name="hidden"/>. Returns false when the kernel cannot handle
        /// the shape (caller falls back to the per-op verify).
        /// </summary>
        public static bool Gemma4MoEModelVerify(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int startPos, int numTokens)
        {
            return GgmlNative.Gemma4MoEModelVerify(layers, numLayers, hidden, hiddenSize, startPos, numTokens);
        }

        public static void Gemma4LayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr postAttnNormW,
            IntPtr ffnNormW,
            IntPtr guW, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downW, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr postFfnNormW,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isLocal, int slidingWindow,
            float ropeBase, int ropeDims,
            IntPtr ropeFreqFactors, int freqFactorsLen,
            float layerScalar, float eps,
            IntPtr swaPrevK = default, IntPtr swaPrevV = default, int prevWindowLen = 0,
            IntPtr pleInputData = default, int pleDim = 0,
            IntPtr pleGateW = default, int pleGateType = 0, long pleGateNe0 = 0, long pleGateNe1 = 0, long pleGateBytes = 0,
            IntPtr pleProjW = default, int pleProjType = 0, long pleProjNe0 = 0, long pleProjNe1 = 0, long pleProjBytes = 0,
            IntPtr plePostNormW = default,
            IntPtr freshKOut = default, IntPtr freshVOut = default,
            int isShared = 0,
            IntPtr donorK = default, IntPtr donorV = default, int donorKvLen = 0,
            int kvCacheType = 0)
        {
            GgmlNative.Gemma4LayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                postAttnNormW,
                ffnNormW,
                guW, guType, guNe0, guNe1, guBytes,
                downW, downType, downNe0, downNe1, downBytes,
                postFfnNormW,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isLocal, slidingWindow,
                ropeBase, ropeDims,
                ropeFreqFactors, freqFactorsLen,
                layerScalar, eps,
                swaPrevK, swaPrevV, prevWindowLen,
                pleInputData, pleDim,
                pleGateW, pleGateType, pleGateNe0, pleGateNe1, pleGateBytes,
                pleProjW, pleProjType, pleProjNe0, pleProjNe1, pleProjBytes,
                plePostNormW,
                freshKOut, freshVOut,
                isShared,
                donorK, donorV, donorKvLen,
                kvCacheType);
        }

        /// <summary>
        /// Fused multi-token prefill attention. Runs Q*K^T → causal mask → softmax → *V as
        /// a single GGML graph on the device (Metal/CUDA/CPU), eliminating ~5 separate C# → GGML
        /// round trips. Supports GQA and optional sliding window.
        ///
        /// Q: [numHeads, seqLen, headDim], K/V: [numKVHeads, kvLen, headDim], all F32 contiguous.
        ///
        /// All four tensors must be Float32: the native kernel hardcodes
        /// <c>GGML_TYPE_F32</c> for the input tensors, so passing a Float16 KV cache (e.g.
        /// the persistent <c>_kvCacheK[layer]</c> on quantized models) silently misreads
        /// half the bytes and produces NaN/garbage output. Callers that may have an F16
        /// cache must dequantize first (e.g. via <c>ExpandKVHeads</c>) or fall back to
        /// the non-fused path.
        /// </summary>
        /// <param name="inputFormat">0 = head-first [numHeads, seqLen, headDim], 1 = flat [seqLen, numHeads*headDim]</param>
        public static void FusedPrefillAttention(Tensor q, Tensor k, Tensor v, Tensor output,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow, float scale, int inputFormat = 0)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (q.ElementType != DType.Float32 || k.ElementType != DType.Float32 ||
                v.ElementType != DType.Float32 || output.ElementType != DType.Float32)
            {
                throw new ArgumentException(
                    $"FusedPrefillAttention requires Float32 tensors but received Q={q.ElementType}, " +
                    $"K={k.ElementType}, V={v.ElementType}, output={output.ElementType}. " +
                    "Dequantize the KV cache (e.g. via ExpandKVHeads) or fall back to the non-fused path.");
            }

            IntPtr qPtr = GetBufferStart(q);
            IntPtr kPtr = GetBufferStart(k);
            IntPtr vPtr = GetBufferStart(v);
            IntPtr outPtr = GetBufferStart(output);

            GgmlNative.FusedPrefillAttention(
                qPtr, kPtr, vPtr, outPtr,
                numHeads, numKvHeads, headDim,
                seqLen, kvLen,
                maskStartPos, slidingWindow, scale, inputFormat);
        }

        /// <summary>
        /// Fused prefill attention that reads K/V straight from an F16 cache (no host
        /// F16→F32 dequant). Q and output are Float32 head-first; <paramref name="kCache"/>
        /// / <paramref name="vCache"/> are the persistent Float16 cache tensors laid out
        /// [numKvHeads, kvCacheLen, headDim], of which the leading <paramref name="kvLen"/>
        /// rows are attended. Numerically identical to dequantizing to F32 first
        /// (mul_mat accumulates in F32), but skips the per-chunk dequant round-trip and
        /// halves the K/V upload. Q is head-first [numHeads, seqLen, headDim].
        /// </summary>
        public static void FusedPrefillAttentionF16KV(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvCacheLen,
            int maskStartPos, int slidingWindow, float scale)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (kCache == null) throw new ArgumentNullException(nameof(kCache));
            if (vCache == null) throw new ArgumentNullException(nameof(vCache));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (q.ElementType != DType.Float32 || output.ElementType != DType.Float32)
                throw new ArgumentException("FusedPrefillAttentionF16KV requires Float32 Q and output.");
            if (kCache.ElementType != DType.Float16 || vCache.ElementType != DType.Float16)
                throw new ArgumentException(
                    $"FusedPrefillAttentionF16KV requires Float16 K/V cache but got K={kCache.ElementType}, V={vCache.ElementType}.");

            IntPtr qPtr = GetBufferStart(q);
            IntPtr kPtr = GetBufferStart(kCache);
            IntPtr vPtr = GetBufferStart(vCache);
            IntPtr outPtr = GetBufferStart(output);

            GgmlNative.FusedPrefillAttentionF16KV(
                qPtr, kPtr, vPtr, outPtr,
                numHeads, numKvHeads, headDim,
                seqLen, kvLen, kvCacheLen,
                maskStartPos, slidingWindow, scale);
        }

        /// <summary>
        /// Single-token Qwen3.5 full-attention decode kernel. Performs the entire FullAttention
        /// block (RMSNorm, fused QKV, deinterleave Q/gate, per-head QK norm, RoPE, KV cache append,
        /// flash attention, sigmoid-gated mix, output projection + residual add) in a single GGML
        /// graph dispatch. Reduces ~2 standalone GGML calls + ~6 small CPU/GPU sync points per
        /// attention layer per decode token to one fused dispatch.
        /// </summary>
        /// <summary>
        /// Fused Qwen3.5 attention layer prefill: full attention block (input
        /// RMSNorm + fused QKV (interleaved Q+gate) + per-head Q/K norm + RoPE
        /// + KV-cache append + causal-masked softmax + attention + sigmoid-gated
        /// mix + output projection + residual add) as a single GGML graph
        /// dispatch per layer.
        ///
        /// Mirrors llama.cpp's src/models/qwen35moe.cpp build and the existing
        /// Qwen35AttentionLayerDecode kernel structure but for multi-row prefill.
        /// </summary>
        public static void Qwen35AttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int ropeMode,
            int kvCacheType, float eps)
        {
            GgmlNative.Qwen35AttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                ropeBase, ropeFreqScale, ropeDims,
                ropeMode, kvCacheType, eps);
        }

        /// <summary>
        /// Fused GPT-OSS attention layer prefill: full attention block (input
        /// RMSNorm + fused QKV (+bias) + RoPE + KV-cache append +
        /// causal/SWA mask + softmax-with-sinks + attention + output projection
        /// (+bias) + residual add) as a single GGML graph dispatch per layer.
        ///
        /// Replaces the ~10 separate per-op submissions on the legacy GPT-OSS
        /// prefill path (each its own Metal command buffer). Mirrors the
        /// llama.cpp graph in src/models/openai-moe-iswa.cpp and the existing
        /// TSGgml_Gemma4LayerPrefill template.
        /// </summary>
        public static void GptOssAttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qkvB,
            int isQkvFused,
            IntPtr kW, int kType, long kNe0, long kNe1, long kBytes,
            IntPtr kB,
            IntPtr vW, int vType, long vNe0, long vNe1, long vBytes,
            IntPtr vB,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr oB,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isSwa, int slidingWindow,
            IntPtr sinksData,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int originalContextLength,
            int kvCacheType, float eps)
        {
            GgmlNative.GptOssAttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes, qkvB, isQkvFused,
                kW, kType, kNe0, kNe1, kBytes, kB,
                vW, vType, vNe0, vNe1, vBytes, vB,
                oW, oType, oNe0, oNe1, oBytes, oB,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isSwa, slidingWindow,
                sinksData,
                ropeBase, ropeFreqScale, ropeDims,
                originalContextLength,
                kvCacheType, eps);
        }

        public static void Qwen35AttentionLayerDecode(
            Tensor residual,
            Tensor attnNorm,
            IntPtr qkvData, int qkvGgmlType, long qkvNe0, long qkvNe1, long qkvRawBytes,
            Tensor qNorm, Tensor kNorm, int headDim,
            IntPtr oData, int oGgmlType, long oNe0, long oNe1, long oRawBytes,
            Tensor kCache, Tensor vCache,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale, int ropeMode)
        {
            if (residual == null || attnNorm == null || qNorm == null || kNorm == null ||
                kCache == null || vCache == null)
                throw new ArgumentNullException("Qwen3.5 attention layer decode requires non-null tensors.");
            if (residual.ElementType != DType.Float32 || attnNorm.ElementType != DType.Float32 ||
                qNorm.ElementType != DType.Float32 || kNorm.ElementType != DType.Float32)
                throw new ArgumentException("Qwen3.5 attention layer decode requires F32 hidden/norm tensors.");
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                throw new ArgumentException("Qwen3.5 attention layer decode requires F32 or F16 K cache tensor.");
            if (vCache.ElementType != kCache.ElementType)
                throw new ArgumentException("Qwen3.5 attention layer decode requires K and V caches to share an element type.");
            if (qkvData == IntPtr.Zero || oData == IntPtr.Zero)
                throw new ArgumentException("Qwen3.5 attention layer decode requires non-zero quantized weight pointers.");

            IntPtr residualPtr = GetBufferStart(residual);
            IntPtr attnNormPtr = GetBufferStart(attnNorm);
            IntPtr qNormPtr = GetBufferStart(qNorm);
            IntPtr kNormPtr = GetBufferStart(kNorm);
            IntPtr kCachePtr = GetBufferStart(kCache);
            IntPtr vCachePtr = GetBufferStart(vCache);
            int hiddenSize = (int)residual.Sizes[residual.Sizes.Length - 1];
            int kvGgmlType = kCache.ElementType == DType.Float16 ? 1 : 0;

            GgmlNative.Qwen35AttentionLayerDecode(
                residualPtr, hiddenSize,
                attnNormPtr,
                qkvData, qkvGgmlType, qkvNe0, qkvNe1, qkvRawBytes,
                qNormPtr, kNormPtr, headDim,
                oData, oGgmlType, oNe0, oNe1, oRawBytes,
                kCachePtr, vCachePtr,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale, ropeMode, kvGgmlType);
        }

        /// <summary>
        /// Fused chunked GatedDeltaNet for Qwen3.5/Qwen3-Next. Runs the prefill recurrent
        /// path entirely on the GGML backend (Metal/CUDA) so the per-token CPU loop in
        /// <see cref="GgmlBasicOps"/> consumers is replaced by a single graph dispatch.
        /// </summary>
        /// <param name="q">Q tensor [seqLen, numVHeads, headDim] (after K-head expansion).</param>
        /// <param name="k">K tensor [seqLen, numVHeads, headDim] (after K-head expansion).</param>
        /// <param name="v">V tensor [seqLen, numVHeads, headDim].</param>
        /// <param name="z">Pre-output gate tensor [seqLen, numVHeads, headDim].</param>
        /// <param name="alpha">Raw alpha [seqLen, numVHeads] (before bias and softplus).</param>
        /// <param name="beta">Raw beta [seqLen, numVHeads] (before sigmoid).</param>
        /// <param name="state">Recurrent state [numVHeads, headDim, headDim], updated in place.</param>
        /// <param name="gatedOut">Output [seqLen, numVHeads, headDim], written into.</param>
        /// <param name="dtBiasData">Per-head bias for alpha [numVHeads].</param>
        /// <param name="aLogData">Per-head -A_log [numVHeads].</param>
        /// <param name="ssmNormWData">Per-D RMSNorm weights [headDim].</param>
        /// <param name="chunkSize">Chunk size (must be a positive power of two).</param>
        /// <param name="eps">Epsilon used for L2Norm and RMSNorm.</param>
        public static void GatedDeltaNetChunked(
            Tensor q, Tensor k, Tensor v, Tensor z,
            Tensor alpha, Tensor beta,
            Tensor state, Tensor gatedOut,
            IntPtr dtBiasData, IntPtr aLogData, IntPtr ssmNormWData,
            int chunkSize, float eps)
        {
            if (q == null || k == null || v == null || z == null
                || alpha == null || beta == null || state == null || gatedOut == null)
                throw new ArgumentNullException(nameof(q), "All tensor arguments must be non-null.");
            if (dtBiasData == IntPtr.Zero || aLogData == IntPtr.Zero || ssmNormWData == IntPtr.Zero)
                throw new ArgumentException("dt_bias, a_log, ssm_norm_w must be non-null pointers.");

            if (!TryCreateStandardView(q, out GgmlTensorView3D qView)
                || !TryCreateStandardView(k, out GgmlTensorView3D kView)
                || !TryCreateStandardView(v, out GgmlTensorView3D vView)
                || !TryCreateStandardView(z, out GgmlTensorView3D zView)
                || !TryCreateStandardView(alpha, out GgmlTensorView2D alphaView)
                || !TryCreateStandardView(beta, out GgmlTensorView2D betaView)
                || !TryCreateStandardView(state, out GgmlTensorView3D stateView)
                || !TryCreateStandardView(gatedOut, out GgmlTensorView3D gatedOutView))
            {
                throw new NotSupportedException("GatedDeltaNetChunked requires Float32 tensors with row-contiguous layouts.");
            }

            GgmlNative.GatedDeltaNetChunked(
                qView, kView, vView, zView,
                alphaView, betaView, stateView, gatedOutView,
                dtBiasData, aLogData, ssmNormWData,
                chunkSize, eps);
        }

        /// <summary>
        /// Batched per-token Nemotron Mamba2 step. Runs all (sequence, token)
        /// pairs in the batched decode/prefill step in one native call,
        /// indexing each sequence's persistent conv FIFO + SSM state via the
        /// <see cref="NemoMamba2BatchedSeqDesc"/> entries.
        /// </summary>
        public static void NemotronMamba2BatchedStep(
            NemoMamba2BatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int dInProjTotal,
            int dInner, int dState, int nHead, int headDim, int nGroup, int dConv,
            IntPtr convWt, IntPtr convBias, IntPtr dtBias, IntPtr aLog,
            IntPtr dData, IntPtr ssmNormW,
            float eps,
            IntPtr outBatched)
        {
            if (seqs == null || seqs.Length == 0)
                throw new ArgumentException("seqs must be non-empty.", nameof(seqs));
            if (packedBatched == IntPtr.Zero || outBatched == IntPtr.Zero)
                throw new ArgumentException("packedBatched and outBatched must be non-null.");

            GgmlNative.NemotronMamba2BatchedStep(
                seqs, numTokens, packedBatched, dInProjTotal,
                dInner, dState, nHead, headDim, nGroup, dConv,
                convWt, convBias, dtBias, aLog, dData, ssmNormW,
                eps, outBatched);
        }

        /// <summary>
        /// Batched per-token Qwen3.5 GatedDeltaNet step. Runs all (sequence,
        /// token) pairs in the batched decode/prefill step in one native call,
        /// indexing each sequence's recurrent conv ring + SSM state via the
        /// <see cref="GdnBatchedSeqDesc"/> entries. The descriptors' ConvWriteIdx
        /// field is updated in place; callers copy it back to per-slot bookkeeping.
        /// </summary>
        public static void GatedDeltaNetBatchedStep(
            GdnBatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int packedDim, int qkvDim, int qkDim, int vDim, int zDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int ssmDInner,
            IntPtr convWt, IntPtr dtBias, IntPtr aLog, IntPtr ssmNormW,
            float eps,
            IntPtr gatedOut)
        {
            if (seqs == null || seqs.Length == 0)
                throw new ArgumentException("seqs must be non-empty.", nameof(seqs));
            if (packedBatched == IntPtr.Zero || gatedOut == IntPtr.Zero)
                throw new ArgumentException("packedBatched and gatedOut must be non-null.");

            GgmlNative.GatedDeltaNetBatchedStep(
                seqs, numTokens,
                packedBatched, packedDim, qkvDim, qkDim, vDim, zDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, ssmDInner,
                convWt, dtBias, aLog, ssmNormW, eps, gatedOut);
        }

        /// <summary>
        /// Native Nemotron Mamba2 prefill core. Runs the causal conv, SSM scan, gate,
        /// and optional group RMSNorm as one GGML graph, updating the recurrent states
        /// in place.
        /// </summary>
        public static unsafe void NemotronMamba2Prefill(
            Tensor projected,
            Tensor hiddenOut,
            float[] convState,
            float[] ssmState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            if (projected == null || hiddenOut == null)
                throw new ArgumentNullException(nameof(projected), "projected and hiddenOut must be non-null.");
            if (convState == null || ssmState == null)
                throw new ArgumentNullException(nameof(convState), "convState and ssmState must be non-null.");
            if (convWeightData == IntPtr.Zero || dtBiasData == IntPtr.Zero || aData == IntPtr.Zero)
                throw new ArgumentException("convWeightData, dtBiasData, and aData must be non-null pointers.");

            if (!TryCreateStandardView(projected, out GgmlTensorView2D projectedView)
                || !TryCreateStandardView(hiddenOut, out GgmlTensorView2D hiddenOutView))
            {
                throw new NotSupportedException("NemotronMamba2Prefill requires Float32 tensors with row-contiguous 2D layouts.");
            }

            fixed (float* convStatePtr = convState)
            fixed (float* ssmStatePtr = ssmState)
            {
                GgmlNative.NemotronMamba2Prefill(
                    projectedView,
                    hiddenOutView,
                    (IntPtr)convStatePtr,
                    convState.Length,
                    (IntPtr)ssmStatePtr,
                    ssmState.Length,
                    convWeightData,
                    convBiasData,
                    dtBiasData,
                    aData,
                    dData,
                    ssmNormData,
                    dInner,
                    dState,
                    nHead,
                    headDim,
                    nGroup,
                    dConv,
                    eps);
            }
        }

        /// <summary>
        /// Native Nemotron Mamba2 single-token decode core. The native side keeps
        /// a persistent recurrent state cache keyed by <paramref name="stateKey"/>;
        /// pass <paramref name="initializeState"/> after reset/prefill to upload
        /// the managed state arrays once.
        /// </summary>
        public static unsafe void NemotronMamba2Decode(
            ulong stateKey,
            Tensor projected,
            Tensor hiddenOut,
            float[] convState,
            float[] ssmState,
            bool initializeState,
            bool downloadState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            if (stateKey == 0)
                throw new ArgumentException("stateKey must be non-zero.", nameof(stateKey));
            if (projected == null || hiddenOut == null)
                throw new ArgumentNullException(nameof(projected), "projected and hiddenOut must be non-null.");
            if (convState == null || ssmState == null)
                throw new ArgumentNullException(nameof(convState), "convState and ssmState must be non-null.");
            if (convWeightData == IntPtr.Zero || dtBiasData == IntPtr.Zero || aData == IntPtr.Zero)
                throw new ArgumentException("convWeightData, dtBiasData, and aData must be non-null pointers.");

            if (!TryCreateStandardView(projected, out GgmlTensorView2D projectedView)
                || !TryCreateStandardView(hiddenOut, out GgmlTensorView2D hiddenOutView))
            {
                throw new NotSupportedException("NemotronMamba2Decode requires Float32 tensors with row-contiguous 2D layouts.");
            }

            fixed (float* convStatePtr = convState)
            fixed (float* ssmStatePtr = ssmState)
            {
                GgmlNative.NemotronMamba2Decode(
                    stateKey,
                    projectedView,
                    hiddenOutView,
                    (IntPtr)convStatePtr,
                    convState.Length,
                    (IntPtr)ssmStatePtr,
                    ssmState.Length,
                    initializeState,
                    downloadState,
                    convWeightData,
                    convBiasData,
                    dtBiasData,
                    aData,
                    dData,
                    ssmNormData,
                    dInner,
                    dState,
                    nHead,
                    headDim,
                    nGroup,
                    dConv,
                    eps);
            }
        }

        public static void NemotronMamba2DecodeClear(ulong modelKey) =>
            GgmlNative.NemotronMamba2DecodeClear(modelKey);

        public static void Gemma4ModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            int[] kvSourceArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int position,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            int kvCacheType = 0,
            IntPtr[] kArr = null, int[] kTypeArr = null, long[] kNe0Arr = null, long[] kNe1Arr = null, long[] kBytesArr = null,
            IntPtr[] vArr = null, int[] vTypeArr = null, long[] vNe0Arr = null, long[] vNe1Arr = null, long[] vBytesArr = null)
        {
            GgmlNative.Gemma4ModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                kvSourceArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, position,
                eps, slidingWindow,
                ropeFreqFactors, ropeFreqFactorsLen,
                ropeNDimsArr,
                pleData, pleDim,
                pleGateArr, pleGateTypeArr, pleGateNe0Arr, pleGateNe1Arr, pleGateBytesArr,
                pleProjArr, pleProjTypeArr, pleProjNe0Arr, pleProjNe1Arr, pleProjBytesArr,
                plePostNormArr, kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr);
        }

        /// <summary>Fused multi-token speculative verify. Returns false when the
        /// native kernel declines (total length past the SWA window) — caller falls
        /// back to the per-op path.</summary>
        public static bool Gemma4ModelVerify(
            IntPtr hiddenData, int hiddenSize, int numLayers, int numTokens,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int startPos, float eps,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen, int[] ropeNDimsArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            int[] kvSourceArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr)
        {
            return GgmlNative.Gemma4ModelVerify(
                hiddenData, hiddenSize, numLayers, numTokens,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, startPos, eps,
                ropeFreqFactors, ropeFreqFactorsLen, ropeNDimsArr,
                kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                kvSourceArr,
                pleData, pleDim,
                pleGateArr, pleGateTypeArr, pleGateNe0Arr, pleGateNe1Arr, pleGateBytesArr,
                pleProjArr, pleProjTypeArr, pleProjNe0Arr, pleProjNe1Arr, pleProjBytesArr,
                plePostNormArr);
        }

        /// <summary>Fused Gemma 4 MTP draft step. Returns false when the native
        /// kernel declines (fixed_pos past the donor SWA window).</summary>
        public static bool Gemma4DraftStep(
            int token, IntPtr hPrev, int fixedPos,
            int backbone, int draftHidden, int numDLayers, int numHeads, int vocab,
            float eps, int kvCacheType,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            IntPtr tgtTokEmbd, int tteType, long tteNe0, long tteNe1, long tteBytes,
            IntPtr nextnPre, int npreType, long npreNe0, long npreNe1, long npreBytes,
            IntPtr nextnPost, int npostType, long npostNe0, long npostNe1, long npostBytes,
            IntPtr draftTokEmbd, int dteType, long dteNe0, long dteNe1, long dteBytes,
            IntPtr outputNormW,
            IntPtr[] attnNormArr, IntPtr[] wqArr, int[] wqType, long[] wqNe0, long[] wqNe1, long[] wqBytes,
            IntPtr[] qNormArr, IntPtr[] woArr, int[] woType, long[] woNe0, long[] woNe1, long[] woBytes,
            IntPtr[] postAttnNormArr, IntPtr[] ffnNormArr,
            IntPtr[] gateArr, int[] gateType, long[] gateNe0, long[] gateNe1, long[] gateBytes,
            IntPtr[] upArr, int[] upType, long[] upNe0, long[] upNe1, long[] upBytes,
            IntPtr[] downArr, int[] downType, long[] downNe0, long[] downNe1, long[] downBytes,
            IntPtr[] postFfwNormArr, float[] outScaleArr,
            int[] hdArr, int[] kvHeadsArr, int[] isLocalArr, float[] ropeBaseArr, int[] ropeDimsArr,
            IntPtr[] donorKArr, IntPtr[] donorVArr, int[] donorCacheSizeArr,
            IntPtr logitsOut, IntPtr hOut)
        {
            return GgmlNative.Gemma4DraftStep(
                token, hPrev, fixedPos,
                backbone, draftHidden, numDLayers, numHeads, vocab,
                eps, kvCacheType,
                ropeFreqFactors, ropeFreqFactorsLen,
                tgtTokEmbd, tteType, tteNe0, tteNe1, tteBytes,
                nextnPre, npreType, npreNe0, npreNe1, npreBytes,
                nextnPost, npostType, npostNe0, npostNe1, npostBytes,
                draftTokEmbd, dteType, dteNe0, dteNe1, dteBytes,
                outputNormW,
                attnNormArr, wqArr, wqType, wqNe0, wqNe1, wqBytes,
                qNormArr, woArr, woType, woNe0, woNe1, woBytes,
                postAttnNormArr, ffnNormArr,
                gateArr, gateType, gateNe0, gateNe1, gateBytes,
                upArr, upType, upNe0, upNe1, upBytes,
                downArr, downType, downNe0, downNe1, downBytes,
                postFfwNormArr, outScaleArr,
                hdArr, kvHeadsArr, isLocalArr, ropeBaseArr, ropeDimsArr,
                donorKArr, donorVArr, donorCacheSizeArr,
                logitsOut, hOut);
        }

        [RegisterOpStorageType("addmmbatch", typeof(GgmlStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            ValidateAddmmBatchArguments(result, src, m1, m2);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView3D srcView = default;
            Tensor compactSrc = null;
            try
            {
                if (!TryCreateStandardView(writeTarget, out GgmlTensorView3D resultView)
                    || (beta != 0.0f && !TryCreateStandardOrBroadcastView(src, out srcView, out compactSrc))
                    || !TryCreateRawView(m1, out GgmlTensorView3D m1View)
                    || !TryCreateRawView(m2, out GgmlTensorView3D m2View))
                {
                    throw new NotSupportedException("GGML addmmbatch requires Float32 tensors with supported row-contiguous/view-compatible layouts.");
                }

                if (beta == 0.0f)
                {
                    srcView = default;
                }

                GgmlNative.AddmmBatch(resultView, srcView, m1View, m2View, beta, alpha);
            }
            finally
            {
                compactSrc?.Dispose();
            }

            return writeTarget;
        }

        [RegisterOpStorageType("mulmatid", typeof(GgmlStorage))]
        public static Tensor MulmatID(Tensor result, Tensor expertWeights, Tensor input, Tensor ids)
        {
            ValidateMulMatIdArguments(result, expertWeights, input, ids);

            long[] outputSizes = new long[] { input.Sizes[0], ids.Sizes[1], expertWeights.Sizes[1] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, expertWeights.Allocator, DType.Float32, false, outputSizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView3D resultView)
                || !TryCreateStandardView(expertWeights, out GgmlTensorView3D expertView)
                || !TryCreateStandardView(input, out GgmlTensorView3D inputView)
                || !TryCreateContiguousTensor(ids, out GgmlContiguousTensor idsTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML mulmatid requires Float32 standard-layout 3D tensors and a contiguous Float32/Int32 id matrix.");
            }

            GgmlNative.MulMatId(resultView, expertView, inputView, idsTensor, (int)ids.Sizes[0], (int)ids.Sizes[1]);
            return writeTarget;
        }

        [RegisterOpStorageType("addid", typeof(GgmlStorage))]
        public static Tensor AddID(Tensor result, Tensor src, Tensor bias, Tensor ids)
        {
            ValidateAddIdArguments(result, src, bias, ids);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView3D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView3D srcView)
                || !TryCreateStandardView(bias, out GgmlTensorView2D biasView)
                || !TryCreateContiguousTensor(ids, out GgmlContiguousTensor idsTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML addid requires Float32 standard-layout source/result tensors, a standard-layout 2D bias matrix, and a contiguous Float32/Int32 id matrix.");
            }

            GgmlNative.AddId(resultView, srcView, biasView, idsTensor, (int)ids.Sizes[0], (int)ids.Sizes[1]);
            return writeTarget;
        }

        [RegisterOpStorageType("softmax", typeof(GgmlStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            ValidateSoftmaxArguments(result, src);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException("GGML softmax requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Softmax(resultView, srcView);
            return writeTarget;
        }

        /// <summary>
        /// In-place fused causal+SWA mask + softmax + optional attention sinks for
        /// the GptOss-style attention path. The <paramref name="scores"/> tensor is
        /// modified in place; its layout must be [numHeads, seqLen, kvLen] (a
        /// 3D Float32 tensor with row-contiguous strides). Pass a null
        /// <paramref name="sinks"/> to disable sinks; pass <paramref name="slidingWindow"/>
        /// &lt;= 0 to disable the sliding window mask.
        ///
        /// This collapses what used to be three operations:
        /// AddCausalMask (GPU) + ApplySWAMask (CPU walk) + ApplySoftmaxWithSinks
        /// (CPU walk) into a single Metal kernel. The CPU softmax-with-sinks loop
        /// was the single largest contributor to GptOss prefill time.
        /// </summary>
        public static unsafe void AttentionSoftmaxWithSinks(
            Tensor scores,
            float[] sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale)
        {
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (!(scores.Storage is GgmlStorage))
                throw new ArgumentException("scores must be a GGML tensor", nameof(scores));
            if (scores.DimensionCount != 3)
                throw new ArgumentException("scores must be 3-D [numHeads, seqLen, kvLen].", nameof(scores));
            if (scores.Sizes[0] != numHeads || scores.Sizes[1] != seqLen || scores.Sizes[2] != kvLen)
                throw new ArgumentException(
                    $"scores shape ({scores.Sizes[0]}, {scores.Sizes[1]}, {scores.Sizes[2]}) doesn't match (numHeads={numHeads}, seqLen={seqLen}, kvLen={kvLen}).",
                    nameof(scores));

            if (!TryCreateStandardView(scores, out GgmlTensorView3D scoresView))
            {
                throw new NotSupportedException(
                    "AttentionSoftmaxWithSinks requires a row-contiguous Float32 scores tensor.");
            }

            if (sinks != null && sinks.Length != numHeads)
                throw new ArgumentException(
                    $"sinks length ({sinks.Length}) doesn't match numHeads ({numHeads}).",
                    nameof(sinks));

            if (sinks == null)
            {
                GgmlNative.AttentionSoftmaxWithSinks(
                    scoresView, IntPtr.Zero, numHeads, seqLen, kvLen,
                    maskStartPos, slidingWindow, scale);
            }
            else
            {
                fixed (float* sinksPtr = sinks)
                {
                    GgmlNative.AttentionSoftmaxWithSinks(
                        scoresView, (IntPtr)sinksPtr, numHeads, seqLen, kvLen,
                        maskStartPos, slidingWindow, scale);
                }
            }
        }

        /// <summary>
        /// Activation mode for <see cref="MoEFFNPrefill"/>. Mirrors
        /// <c>MoEActivation</c> in <c>ggml_ops_moe.cpp</c> so C# callers can
        /// select the GLU variant without poking ints through the ABI.
        /// </summary>
        public enum MoEActivation : int
        {
            /// <summary>silu(gate) * up (Qwen 3.5 / Mixtral / generic MoE).</summary>
            SwiGLUSplit = 0,
            /// <summary>GPT-OSS clamped SwiGLU variant (uses alpha/limit).</summary>
            SwiGLUOAI = 1,
            /// <summary>gelu(gate) * up tanh-approx GeGLU (Gemma 4 MoE).</summary>
            GEGLUSplit = 2,
            /// <summary>relu(up)^2 single-projection FFN (Nemotron-H MoE).</summary>
            ReluSquared = 3,
        }

        /// <summary>
        /// Fused MoE FFN forward: expert projection(s), activation
        /// (SwiGLU split / SwiGLU OAI / GeGLU split / ReLU-squared), down projection,
        /// per-token expert weighting, and cross-expert aggregation in a
        /// single GGML graph dispatch (one <c>ggml_mul_mat_id</c> per matmul,
        /// matching llama.cpp's <c>build_moe_ffn</c>).
        ///
        /// This collapses the previous per-active-expert loop (which issued
        /// O(num_active_experts × num_layers) GGML graph submissions per ubatch)
        /// to a constant 1 submission per layer regardless of how many experts
        /// are active. For GPT-OSS (32 experts × 4 active × ~24 layers), Qwen
        /// 3.5 MoE (128 × 8), and Gemma 4 26B-A4B MoE (16 × 4) this is the
        /// dominant MoE decode/prefill cost.
        ///
        /// Weight tensors must be passed as the original GGUF stacked layout
        /// <c>[ne0, ne1, num_experts]</c> via a base host pointer + total
        /// bytes. For mmap'd models the C# layer can pass the base pointer of
        /// expert 0 directly (it points to the contiguous block); for non-mmap
        /// models the caller stacks experts into a single buffer once at model
        /// load time.
        ///
        /// Set <paramref name="upData"/> to <see cref="IntPtr.Zero"/> when
        /// <paramref name="gateData"/> already contains a pre-fused gate+up
        /// weight (gate_ne1 = 2*nFf), mirroring llama.cpp's gate_up_exps path,
        /// or when <paramref name="activation"/> is <see cref="MoEActivation.ReluSquared"/>
        /// and <paramref name="gateData"/> is the single up projection.
        ///
        /// <paramref name="hiddenIn"/> and <paramref name="hiddenOut"/> must
        /// both be row-contiguous Float32 tensors of shape
        /// <c>[seqLen, hiddenDim]</c>. The kernel writes only to
        /// <paramref name="hiddenOut"/>; the caller is responsible for adding
        /// it back to a residual.
        ///
        /// Per-expert post-down-projection scales (Gemma 4's
        /// <c>ffn_down_exps.scale</c>) should be folded into
        /// <paramref name="routingWeights"/> by the caller before the call.
        /// </summary>
        public static unsafe void MoEFFNPrefill(
            Tensor hiddenIn,
            Tensor hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            int[] selectedExperts,
            float[] routingWeights,
            IntPtr gateData, int gateGgmlType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upGgmlType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downTotalBytes,
            float[] gateBias,         // null to skip; otherwise [biasDim, numExperts] (biasDim = nFf or 2*nFf for fused gate_up)
            float[] upBias,           // null to skip; only valid when upData != IntPtr.Zero
            float[] downBias,         // null to skip; otherwise [hiddenDim, numExperts]
            MoEActivation activation,
            float oaiAlpha = 1.702f,
            float oaiLimit = 7.0f)
        {
            if (hiddenIn == null) throw new ArgumentNullException(nameof(hiddenIn));
            if (hiddenOut == null) throw new ArgumentNullException(nameof(hiddenOut));
            if (selectedExperts == null) throw new ArgumentNullException(nameof(selectedExperts));
            if (routingWeights == null) throw new ArgumentNullException(nameof(routingWeights));

            if (!(hiddenIn.Storage is GgmlStorage))
                throw new ArgumentException("hiddenIn must be a GGML tensor.", nameof(hiddenIn));
            if (!(hiddenOut.Storage is GgmlStorage))
                throw new ArgumentException("hiddenOut must be a GGML tensor.", nameof(hiddenOut));

            if (hiddenIn.ElementType != DType.Float32 || hiddenOut.ElementType != DType.Float32)
                throw new ArgumentException("hidden tensors must be Float32.");
            if (hiddenIn.DimensionCount != 2 || hiddenOut.DimensionCount != 2)
                throw new ArgumentException("hidden tensors must be 2D [seqLen, hiddenDim].");
            if (hiddenIn.Sizes[0] != seqLen || hiddenIn.Sizes[1] != hiddenDim ||
                hiddenOut.Sizes[0] != seqLen || hiddenOut.Sizes[1] != hiddenDim)
                throw new ArgumentException("hidden tensor shape doesn't match (seqLen, hiddenDim).");
            if (!hiddenIn.IsContiguous() || !hiddenOut.IsContiguous())
                throw new ArgumentException("hidden tensors must be contiguous.");

            // Length check is >= rather than == so callers can pass pooled
            // scratch buffers sized for the largest expected seqLen and reuse
            // them across decode steps. The kernel only reads the first
            // seqLen * nUsed entries.
            if (selectedExperts.Length < seqLen * nUsed)
                throw new ArgumentException($"selectedExperts length {selectedExperts.Length} < seqLen*nUsed = {seqLen * nUsed}.");
            if (routingWeights.Length < seqLen * nUsed)
                throw new ArgumentException($"routingWeights length {routingWeights.Length} < seqLen*nUsed = {seqLen * nUsed}.");

            if (gateData == IntPtr.Zero || downData == IntPtr.Zero)
                throw new ArgumentException("gate and down weight pointers must be non-null.");

            bool reluSquared = activation == MoEActivation.ReluSquared;
            bool fusedGateUp = upData == IntPtr.Zero && !reluSquared;
            if (reluSquared && upData != IntPtr.Zero)
                throw new ArgumentException("ReluSquared MoE uses a single projection; pass it as gateData and leave upData as IntPtr.Zero.");
            int gateBiasDim = fusedGateUp ? 2 * nFf : nFf;
            if (gateBias != null && gateBias.Length != gateBiasDim * numExperts)
                throw new ArgumentException($"gateBias length {gateBias.Length} != gateBiasDim * numExperts = {gateBiasDim * numExperts}.");
            if (upBias != null)
            {
                if (fusedGateUp || reluSquared) throw new ArgumentException("upBias must be null when upData is null.");
                if (upBias.Length != nFf * numExperts)
                    throw new ArgumentException($"upBias length {upBias.Length} != nFf * numExperts = {nFf * numExperts}.");
            }
            if (downBias != null && downBias.Length != hiddenDim * numExperts)
                throw new ArgumentException($"downBias length {downBias.Length} != hiddenDim * numExperts = {hiddenDim * numExperts}.");

            // hiddenIn/Out are GGML host-mapped (zero-copy on Metal) so we can
            // pass their raw storage pointers. EnsureHostReadable in case there
            // are pending GPU writes from a previous async op upstream.
            hiddenIn.Storage.EnsureHostReadable();
            hiddenOut.Storage.EnsureHostReadable();

            IntPtr hiddenInPtr = TensorComputePrimitives.GetStoragePointer(hiddenIn);
            IntPtr hiddenOutPtr = TensorComputePrimitives.GetStoragePointer(hiddenOut);

            int activationType = (int)activation;

            fixed (int* idsPtr = selectedExperts)
            fixed (float* weightsPtr = routingWeights)
            fixed (float* gateBiasPtr = gateBias)
            fixed (float* upBiasPtr = upBias)
            fixed (float* downBiasPtr = downBias)
            {
                GgmlNative.MoEFFNPrefillSwiGLUQuant(
                    hiddenInPtr, hiddenOutPtr, seqLen, hiddenDim, nFf,
                    numExperts, nUsed,
                    (IntPtr)idsPtr, (IntPtr)weightsPtr,
                    gateData, gateGgmlType, gateNe0, gateNe1, gateTotalBytes,
                    upData,   upGgmlType,   upNe0,   upNe1,   upTotalBytes,
                    downData, downGgmlType, downNe0, downNe1, downTotalBytes,
                    gateBias != null ? (IntPtr)gateBiasPtr : IntPtr.Zero,
                    upBias   != null ? (IntPtr)upBiasPtr   : IntPtr.Zero,
                    downBias != null ? (IntPtr)downBiasPtr : IntPtr.Zero,
                    activationType, oaiAlpha, oaiLimit);
            }
        }

        /// <summary>
        /// Gemma 4 MoE GEGLU + post_ffw_norm_2 + residual-add fused kernel.
        ///
        /// Fuses the entire post-attention MoE FFN block of a Gemma 4 MoE
        /// layer into a single GGML graph dispatch:
        /// <code>
        ///   moe_out      = MoE_FFN(hidden_in, gate, up, down, ids, weights)
        ///   moe_normed   = rms_norm(moe_out, eps) * post_ffw_norm_2.weight
        ///   residual    += moe_normed
        /// </code>
        /// versus the previous unfused sequence which issued three separate
        /// dispatches (TryMoEFusedGEGLU + Ops.RMSNorm + Ops.Add). On Apple
        /// Silicon Metal each per-op dispatch costs ~50 µs of submission
        /// latency, so collapsing two of them per MoE layer saves on the
        /// order of 100 µs × num_moe_layers per token. This is the Gemma 4
        /// analogue of <see cref="MoEExpertsSwiGLUResidual"/> for Qwen 3.5.
        ///
        /// <paramref name="residual"/> must be a row-contiguous Float32
        /// <c>[seqLen, hiddenDim]</c> tensor that already contains the value
        /// to add the normed MoE output to (typically the dense FFN output
        /// after Gemma 4's <c>post_ffw_norm_1</c>). The kernel writes the sum
        /// back over the same backend buffer in place.
        ///
        /// Same weight-layout / activation-mode rules as
        /// <see cref="MoEFFNPrefill"/>; the routed-expert per-down-projection
        /// scales (Gemma 4's <c>ffn_down_exps.scale</c>) must be folded into
        /// <paramref name="routingWeights"/> by the caller.
        /// </summary>
        public static unsafe void MoEFFNGEGLUResidualGemma4(
            Tensor hiddenIn,
            Tensor residual,
            Tensor postNormW,
            float postNormEps,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            int[] selectedExperts,
            float[] routingWeights,
            IntPtr gateData, int gateGgmlType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upGgmlType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downTotalBytes,
            float[] gateBias,
            float[] upBias,
            float[] downBias,
            MoEActivation activation = MoEActivation.GEGLUSplit,
            float oaiAlpha = 1.702f,
            float oaiLimit = 7.0f)
        {
            if (hiddenIn == null) throw new ArgumentNullException(nameof(hiddenIn));
            if (residual == null) throw new ArgumentNullException(nameof(residual));
            if (postNormW == null) throw new ArgumentNullException(nameof(postNormW));
            if (selectedExperts == null) throw new ArgumentNullException(nameof(selectedExperts));
            if (routingWeights == null) throw new ArgumentNullException(nameof(routingWeights));

            if (!(hiddenIn.Storage is GgmlStorage))
                throw new ArgumentException("hiddenIn must be a GGML tensor.", nameof(hiddenIn));
            if (!(residual.Storage is GgmlStorage))
                throw new ArgumentException("residual must be a GGML tensor.", nameof(residual));
            if (!(postNormW.Storage is GgmlStorage))
                throw new ArgumentException("postNormW must be a GGML tensor.", nameof(postNormW));

            if (hiddenIn.ElementType != DType.Float32 || residual.ElementType != DType.Float32 ||
                postNormW.ElementType != DType.Float32)
                throw new ArgumentException("hiddenIn / residual / postNormW must be Float32.");
            if (hiddenIn.DimensionCount != 2 || residual.DimensionCount != 2)
                throw new ArgumentException("hiddenIn and residual must be 2D [seqLen, hiddenDim].");
            if (hiddenIn.Sizes[0] != seqLen || hiddenIn.Sizes[1] != hiddenDim ||
                residual.Sizes[0] != seqLen || residual.Sizes[1] != hiddenDim)
                throw new ArgumentException("tensor shape doesn't match (seqLen, hiddenDim).");
            if (postNormW.ElementCount() != hiddenDim)
                throw new ArgumentException($"postNormW must have hiddenDim elements (got {postNormW.ElementCount()}).");
            if (!hiddenIn.IsContiguous() || !residual.IsContiguous())
                throw new ArgumentException("hiddenIn and residual must be contiguous.");

            // Length check is >= rather than == so callers can pass pooled
            // scratch buffers sized for the largest expected seqLen and reuse
            // them across decode steps. The kernel only reads the first
            // seqLen * nUsed entries.
            if (selectedExperts.Length < seqLen * nUsed)
                throw new ArgumentException($"selectedExperts length {selectedExperts.Length} < seqLen*nUsed = {seqLen * nUsed}.");
            if (routingWeights.Length < seqLen * nUsed)
                throw new ArgumentException($"routingWeights length {routingWeights.Length} < seqLen*nUsed = {seqLen * nUsed}.");

            if (gateData == IntPtr.Zero || downData == IntPtr.Zero)
                throw new ArgumentException("gate and down weight pointers must be non-null.");

            bool reluSquared = activation == MoEActivation.ReluSquared;
            bool fusedGateUp = upData == IntPtr.Zero && !reluSquared;
            if (reluSquared && upData != IntPtr.Zero)
                throw new ArgumentException("ReluSquared MoE uses a single projection; pass it as gateData and leave upData as IntPtr.Zero.");
            int gateBiasDim = fusedGateUp ? 2 * nFf : nFf;
            if (gateBias != null && gateBias.Length != gateBiasDim * numExperts)
                throw new ArgumentException($"gateBias length {gateBias.Length} != gateBiasDim * numExperts = {gateBiasDim * numExperts}.");
            if (upBias != null)
            {
                if (fusedGateUp || reluSquared) throw new ArgumentException("upBias must be null when upData is null.");
                if (upBias.Length != nFf * numExperts)
                    throw new ArgumentException($"upBias length {upBias.Length} != nFf * numExperts = {nFf * numExperts}.");
            }
            if (downBias != null && downBias.Length != hiddenDim * numExperts)
                throw new ArgumentException($"downBias length {downBias.Length} != hiddenDim * numExperts = {hiddenDim * numExperts}.");

            // Drain pending GPU writes targeting the residual / hidden / norm
            // weight buffers so the kernel sees an up-to-date view.
            hiddenIn.Storage.EnsureHostReadable();
            residual.Storage.EnsureHostReadable();
            postNormW.Storage.EnsureHostReadable();

            IntPtr hiddenInPtr = TensorComputePrimitives.GetStoragePointer(hiddenIn);
            IntPtr residualPtr = TensorComputePrimitives.GetStoragePointer(residual);
            IntPtr postNormPtr = TensorComputePrimitives.GetStoragePointer(postNormW);

            int activationType = (int)activation;

            fixed (int* idsPtr = selectedExperts)
            fixed (float* weightsPtr = routingWeights)
            fixed (float* gateBiasPtr = gateBias)
            fixed (float* upBiasPtr = upBias)
            fixed (float* downBiasPtr = downBias)
            {
                GgmlNative.Gemma4MoEGEGLUResidual(
                    hiddenInPtr, residualPtr, postNormPtr, postNormEps,
                    seqLen, hiddenDim, nFf,
                    numExperts, nUsed,
                    (IntPtr)idsPtr, (IntPtr)weightsPtr,
                    gateData, gateGgmlType, gateNe0, gateNe1, gateTotalBytes,
                    upData,   upGgmlType,   upNe0,   upNe1,   upTotalBytes,
                    downData, downGgmlType, downNe0, downNe1, downTotalBytes,
                    gateBias != null ? (IntPtr)gateBiasPtr : IntPtr.Zero,
                    upBias   != null ? (IntPtr)upBiasPtr   : IntPtr.Zero,
                    downBias != null ? (IntPtr)downBiasPtr : IntPtr.Zero,
                    activationType, oaiAlpha, oaiLimit);
            }
        }

        /// <summary>
        /// Backwards-compatible wrapper for <see cref="MoEFFNPrefill"/> that
        /// selects SwiGLU (or SwiGLU OAI) activation. Retained so existing
        /// Qwen 3.5 / GPT-OSS call sites don't need to care about the broader
        /// activation enum.
        /// </summary>
        public static unsafe void MoEFFNPrefillSwiGLU(
            Tensor hiddenIn,
            Tensor hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            int[] selectedExperts,
            float[] routingWeights,
            IntPtr gateData, int gateGgmlType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upGgmlType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downTotalBytes,
            float[] gateBias,
            float[] upBias,
            float[] downBias,
            bool useSwiGLUOAI,
            float oaiAlpha = 1.702f,
            float oaiLimit = 7.0f)
        {
            MoEFFNPrefill(
                hiddenIn, hiddenOut, seqLen, hiddenDim, nFf, numExperts, nUsed,
                selectedExperts, routingWeights,
                gateData, gateGgmlType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upGgmlType,   upNe0,   upNe1,   upTotalBytes,
                downData, downGgmlType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                useSwiGLUOAI ? MoEActivation.SwiGLUOAI : MoEActivation.SwiGLUSplit,
                oaiAlpha, oaiLimit);
        }

        [RegisterOpStorageType("scaled_dot_product_attention", typeof(GgmlStorage))]
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            ValidateScaledDotProductAttentionArguments(result, query, key, value, mask);

            long[] outputSizes = new long[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, query.ElementType, false, outputSizes);
            GgmlTensorView4D maskView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(query, out GgmlTensorView4D queryView)
                || !TryCreateStandardView(key, out GgmlTensorView4D keyView)
                || !TryCreateStandardView(value, out GgmlTensorView4D valueView)
                || (mask != null && !TryCreateStandardView(mask, out maskView)))
            {
                throw new NotSupportedException("GGML scaled_dot_product_attention requires Float32 tensors with standard 4D layouts.");
            }

            GgmlNative.ScaledDotProductAttention(resultView, queryView, keyView, valueView, maskView, mask != null, scale);
            return writeTarget;
        }

        [RegisterOpStorageType("softmaxgrad", typeof(GgmlStorage))]
        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            ValidateSoftmaxGradArguments(grad, adj, val);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad, adj, false, adj.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView)
                || !TryCreateStandardView(val, out GgmlTensorView4D valView))
            {
                throw new NotSupportedException("GGML softmaxgrad requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.SoftmaxGrad(resultView, adjView, valView, addGrad);
            return writeTarget;
        }

        [RegisterOpStorageType("adam", typeof(GgmlStorage))]
        public static Tensor Adam(
            Tensor tw,
            Tensor tg,
            Tensor tv,
            Tensor tm,
            float gradNormFactor,
            float stepSize,
            float clipval,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            ValidateAdamArguments(tw, tg, tv, tm);

            if (!TryCreateContiguousTensor(tw, out GgmlContiguousTensor weight)
                || !TryCreateContiguousTensor(tg, out GgmlContiguousTensor gradient)
                || !TryCreateContiguousTensor(tv, out GgmlContiguousTensor v)
                || !TryCreateContiguousTensor(tm, out GgmlContiguousTensor m))
            {
                throw new NotSupportedException("GGML Adam requires contiguous Float32 tensors.");
            }

            GgmlNative.Adam(weight, gradient, v, m, gradNormFactor, stepSize, clipval, regc, decayRateV, decayRateM, iter, eps);
            return tw;
        }

        [RegisterOpStorageType("copy", typeof(GgmlStorage))]
        public static unsafe void Copy(Tensor result, Tensor src)
        {
            ValidateCopyArguments(result, src);

            long elementCount = result.ElementCount();
            if (elementCount == 0)
            {
                return;
            }

            DType dtype = result.ElementType;
            byte* resultBuffer = (byte*)GetBufferStart(result);
            byte* srcBuffer = (byte*)GetBufferStart(src);

            if (result.IsContiguous() && src.IsContiguous())
            {
                long byteCount = dtype.ByteLengthFor(elementCount);
                Buffer.MemoryCopy(srcBuffer, resultBuffer, byteCount, byteCount);
                return;
            }

            // Float32 keeps the historical element-level iteration: callers
            // build ad-hoc strided F32 tensors where strides aren't guaranteed
            // to align to anything coarser than one float, so we walk element
            // by element via TensorIterState's float* cursor.
            if (dtype == DType.Float32)
            {
                TensorIterState resultIter = new TensorIterState((float*)resultBuffer, result.DimensionCount, result.SizesMemory, result.StridesMemory);
                TensorIterState srcIter = new TensorIterState((float*)srcBuffer, src.DimensionCount, src.SizesMemory, src.StridesMemory);

                do
                {
                    for (; !resultIter.ReachedBlockEnd() && !srcIter.ReachedBlockEnd(); resultIter.BlockStep(), srcIter.BlockStep())
                    {
                        *resultIter.data = *srcIter.data;
                    }
                } while (resultIter.NextBlock() && srcIter.NextBlock());
                return;
            }

            // Strided Float16 / Q8_0 fallback: copy the largest inner contiguous
            // block as raw bytes per outer index. KV-cache resize narrows a
            // [heads, capacity, head_dim] tensor on the token dim, so each
            // head's _cacheSeqLen * head_dim slice is contiguous in memory and
            // gets memcpy'd in one shot; the outer walk just iterates heads.
            // For Q8_0 the inner extent must align to the 32-element block
            // boundary so byte offsets stay block-aligned.
            CopyStridedBytes(result, src, resultBuffer, srcBuffer, dtype);
        }

        private static unsafe void CopyStridedBytes(Tensor result, Tensor src, byte* resultBuffer, byte* srcBuffer, DType dtype)
        {
            int dimCount = result.DimensionCount;
            ReadOnlySpan<long> resultSizes = result.Sizes;
            ReadOnlySpan<long> resultStrides = result.Strides;
            ReadOnlySpan<long> srcStrides = src.Strides;

            long innerElems = 1;
            int outerDims = dimCount;
            for (int d = dimCount - 1; d >= 0; d--)
            {
                if (resultStrides[d] != innerElems || srcStrides[d] != innerElems)
                    break;
                innerElems *= resultSizes[d];
                outerDims = d;
            }

            if (outerDims == dimCount)
            {
                throw new NotSupportedException(
                    $"copy of strided {dtype} tensors requires the innermost dimension to be contiguous on both sides.");
            }

            if (dtype == DType.Q8_0 && (innerElems % 32) != 0)
            {
                throw new NotSupportedException(
                    $"copy of strided Q8_0 tensors requires the inner contiguous extent ({innerElems} elements) to be a multiple of 32.");
            }

            long innerBytes = dtype.ByteLengthFor(innerElems);

            long[] counter = outerDims > 0 ? new long[outerDims] : System.Array.Empty<long>();
            while (true)
            {
                long resultElemOffset = 0;
                long srcElemOffset = 0;
                for (int d = 0; d < outerDims; d++)
                {
                    resultElemOffset += counter[d] * resultStrides[d];
                    srcElemOffset += counter[d] * srcStrides[d];
                }

                long resultByteOffset = ElementOffsetToBytes(resultElemOffset, dtype);
                long srcByteOffset = ElementOffsetToBytes(srcElemOffset, dtype);
                Buffer.MemoryCopy(srcBuffer + srcByteOffset, resultBuffer + resultByteOffset, innerBytes, innerBytes);

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
                    return;
            }
        }

        private static long ElementOffsetToBytes(long elementOffset, DType dtype)
        {
            if (dtype == DType.Q8_0)
            {
                // Q8_0 stores 32 elements per 34-byte block; outer strides feeding
                // into this helper are always multiples of 32 elements (validated
                // by the inner-extent alignment check above for the typical KV
                // cache layout [heads, capacity, head_dim] where head_dim % 32 == 0).
                if ((elementOffset % 32) != 0)
                    throw new InvalidOperationException(
                        $"Q8_0 byte offset requires element offset ({elementOffset}) to align to 32-element blocks.");
                return (elementOffset / 32) * 34;
            }
            return elementOffset * dtype.Size();
        }

        [RegisterOpStorageType("sum", typeof(GgmlStorage))]
        public static unsafe Tensor Sum(Tensor result, Tensor src, int dimension) => ExecuteReduction(result, src, dimension, false, "sum");

        [RegisterOpStorageType("mean", typeof(GgmlStorage))]
        public static unsafe Tensor Mean(Tensor result, Tensor src, int dimension) => ExecuteReduction(result, src, dimension, true, "mean");

        [RegisterOpStorageType("argmin", typeof(GgmlStorage))]
        public static unsafe Tensor Argmin(Tensor result, Tensor src, int dimension) => ExecuteIndexReduction(result, src, dimension, true, "argmin");

        [RegisterOpStorageType("argmax", typeof(GgmlStorage))]
        public static unsafe Tensor Argmax(Tensor result, Tensor src, int dimension) => ExecuteIndexReduction(result, src, dimension, false, "argmax");

        [RegisterOpStorageType("topK", typeof(GgmlStorage))]
        public static void TopK(Tensor outVal, Tensor outIdx, Tensor src, int k)
        {
            ValidateGgmlTensor(src, nameof(src), "topK");
            ValidateGgmlTensor(outVal, nameof(outVal), "topK");
            ValidateGgmlTensor(outIdx, nameof(outIdx), "topK");

            if (k <= 0 || k > src.Sizes[^1])
            {
                throw new ArgumentOutOfRangeException(nameof(k), "topK requires 0 < k <= last dimension.");
            }

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];
            if (storageSize % cols != 0)
            {
                throw new InvalidOperationException("topK expects a tensor that can be flattened into full rows along the last dimension.");
            }

            long rows = storageSize / cols;
            if (outVal.Sizes.Length != src.Sizes.Length || outIdx.Sizes.Length != src.Sizes.Length)
            {
                throw new InvalidOperationException("topK expects outVal/outIdx to have the same rank as src.");
            }

            for (int i = 0; i < src.Sizes.Length - 1; i++)
            {
                if (outVal.Sizes[i] != src.Sizes[i] || outIdx.Sizes[i] != src.Sizes[i])
                {
                    throw new InvalidOperationException("topK expects outVal/outIdx to match src in every dimension except the last.");
                }
            }

            if (outVal.Sizes[^1] != k || outIdx.Sizes[^1] != k)
            {
                throw new InvalidOperationException("topK expects the last output dimension to equal k.");
            }

            float[] input = src.GetElementsAsFloat((int)src.ElementCount());
            float[] values = new float[rows * k];
            float[] indices = new float[rows * k];

            for (int row = 0; row < rows; row++)
            {
                int rowOffset = checked((int)(row * cols));
                int outOffset = checked((int)(row * k));
                for (int i = 0; i < k; i++)
                {
                    values[outOffset + i] = float.NegativeInfinity;
                    indices[outOffset + i] = -1.0f;
                }

                for (int col = 0; col < cols; col++)
                {
                    float candidate = input[rowOffset + col];
                    for (int slot = 0; slot < k; slot++)
                    {
                        if (candidate <= values[outOffset + slot])
                        {
                            continue;
                        }

                        for (int shift = k - 1; shift > slot; shift--)
                        {
                            values[outOffset + shift] = values[outOffset + shift - 1];
                            indices[outOffset + shift] = indices[outOffset + shift - 1];
                        }

                        values[outOffset + slot] = candidate;
                        indices[outOffset + slot] = col;
                        break;
                    }
                }
            }

            outVal.CopyFrom(values);
            outIdx.CopyFrom(indices);
        }

        [RegisterOpStorageType("iscorrupted", typeof(GgmlStorage))]
        public static unsafe bool IsCorrupted(Tensor src)
        {
            ValidateGgmlTensor(src, nameof(src), "iscorrupted");

            float* buffer = (float*)GetBufferStart(src);
            TensorIterState iter = new TensorIterState(buffer, src.DimensionCount, src.SizesMemory, src.StridesMemory);
            do
            {
                for (; !iter.ReachedBlockEnd(); iter.BlockStep())
                {
                    if (!float.IsFinite(*iter.data))
                    {
                        return true;
                    }
                }
            } while (iter.NextBlock());

            return false;
        }

        [RegisterOpStorageType("abs", typeof(GgmlStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Abs, "abs");

        [RegisterOpStorageType("neg", typeof(GgmlStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Neg, "neg");

        [RegisterOpStorageType("sqrt", typeof(GgmlStorage))]
        public static Tensor Sqrt(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Sqrt, "sqrt");

        [RegisterOpStorageType("exp", typeof(GgmlStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Exp, "exp");

        [RegisterOpStorageType("log", typeof(GgmlStorage))]
        public static Tensor Log(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Log, "log");

        [RegisterOpStorageType("relu", typeof(GgmlStorage))]
        public static Tensor Relu(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Relu, "relu");

        [RegisterOpStorageType("sigmoid", typeof(GgmlStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Sigmoid, "sigmoid");

        [RegisterOpStorageType("tanh", typeof(GgmlStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Tanh, "tanh");

        [RegisterOpStorageType("SiLU", typeof(GgmlStorage))]
        public static Tensor SiLU(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.SiLU, "SiLU");

        [RegisterOpStorageType("GELU", typeof(GgmlStorage))]
        public static Tensor GELU(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.GELU, "GELU");

        [RegisterOpStorageType("SiLUMul", typeof(GgmlStorage))]
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up) => ExecuteFusedActMul(result, gate, up, GgmlFusedActMulOp.SiLUMul, "SiLUMul");

        /// <summary>
        /// Fused SiLU(gate) * up where gate and up are stored as the two halves of a single
        /// contiguous [N, 2H] gate_up tensor. Avoids the two large NewContiguous copies that
        /// the split + 2-arg path requires for prefill (seqLen &gt; 1).
        /// gateUp must be Float32 with shape [N, 2*halfDim] and a row-major layout (stride1 == 1).
        /// Result is allocated/written as [N, halfDim].
        /// </summary>
        [RegisterOpStorageType("SiLUMulSplit", typeof(GgmlStorage))]
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
            => ExecuteFusedActMulSplit(result, gateUp, halfDim, GgmlFusedActMulOp.SiLUMul, "SiLUMulSplit");

        [RegisterOpStorageType("GELUMul", typeof(GgmlStorage))]
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up) => ExecuteFusedActMul(result, gate, up, GgmlFusedActMulOp.GELUMul, "GELUMul");

        [RegisterOpStorageType("SigmoidMul", typeof(GgmlStorage))]
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate) => ExecuteFusedActMul(result, x, gate, GgmlFusedActMulOp.SigmoidMul, "SigmoidMul");

        [RegisterOpStorageType("addt", typeof(GgmlStorage))]
        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Add, "addt");

        [RegisterOpStorageType("subt", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Sub, "subt");

        [RegisterOpStorageType("mult", typeof(GgmlStorage))]
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Mul, "mult");

        [RegisterOpStorageType("divt", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Div, "divt");

        [RegisterOpStorageType("addv", typeof(GgmlStorage))]
        public static Tensor Add(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Add, "addv");

        [RegisterOpStorageType("subv", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Sub, "subv");

        [RegisterOpStorageType("rsubv", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) => ExecuteBinaryScalar(result, rhs, lhs, GgmlBinaryScalarOp.ReverseSub, "rsubv");

        [RegisterOpStorageType("mulv", typeof(GgmlStorage))]
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Mul, "mulv");

        [RegisterOpStorageType("divv", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Div, "divv");

        [RegisterOpStorageType("rdivv", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) => ExecuteBinaryScalar(result, rhs, lhs, GgmlBinaryScalarOp.ReverseDiv, "rdivv");

        [RegisterOpStorageType("addmul", typeof(GgmlStorage))]
        public static unsafe Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            ValidateElementwiseArguments(result, "addmul", x, y, z);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.SizesMemory, x.StridesMemory);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.SizesMemory, y.StridesMemory);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.SizesMemory, z.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data * *zIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("adddiv", typeof(GgmlStorage))]
        public static unsafe Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            ValidateElementwiseArguments(result, "adddiv", x, y, z);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.SizesMemory, x.StridesMemory);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.SizesMemory, y.StridesMemory);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.SizesMemory, z.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data / *zIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("addmulv", typeof(GgmlStorage))]
        public static unsafe Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            ValidateElementwiseArguments(result, "addmulv", x, y);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.SizesMemory, x.StridesMemory);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.SizesMemory, y.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data * z);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("mulmuladd", typeof(GgmlStorage))]
        public static unsafe Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w)
        {
            ValidateElementwiseArguments(result, "mulmuladd", x, y, z, w);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.SizesMemory, x.StridesMemory);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.SizesMemory, y.StridesMemory);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.SizesMemory, z.StridesMemory);
            TensorIterState wIter = new TensorIterState((float*)GetBufferStart(w), w.DimensionCount, w.SizesMemory, w.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd() && !wIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep(), wIter.BlockStep())
                {
                    *resultIter.data = (*xIter.data * *yIter.data) + (*zIter.data * *wIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock() && wIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("relud", typeof(GgmlStorage))]
        public static Tensor ReluD(Tensor result, Tensor w, Tensor g) => ExecuteActivationGrad(result, null, w, g, GgmlActivationGradOp.Relu, "relud");

        [RegisterOpStorageType("addrelud", typeof(GgmlStorage))]
        public static Tensor AddReluD(Tensor result, Tensor src, Tensor w, Tensor g) => ExecuteActivationGrad(result, src, w, g, GgmlActivationGradOp.Relu, "addrelud");

        [RegisterOpStorageType("sigmoidD", typeof(GgmlStorage))]
        public static Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, null, resW, resG, GgmlActivationGradOp.Sigmoid, "sigmoidD");

        [RegisterOpStorageType("addsigmoidD", typeof(GgmlStorage))]
        public static Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, t, resW, resG, GgmlActivationGradOp.Sigmoid, "addsigmoidD");

        [RegisterOpStorageType("tanhD", typeof(GgmlStorage))]
        public static Tensor TanhD(Tensor result, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, null, resW, resG, GgmlActivationGradOp.Tanh, "tanhD");

        [RegisterOpStorageType("addtanhD", typeof(GgmlStorage))]
        public static Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, t, resW, resG, GgmlActivationGradOp.Tanh, "addtanhD");

        [RegisterOpStorageType("SiLUD", typeof(GgmlStorage))]
        public static Tensor SiLUD(Tensor result, Tensor srcW, Tensor resG) => ExecuteActivationGrad(result, null, srcW, resG, GgmlActivationGradOp.SiLU, "SiLUD");

        [RegisterOpStorageType("AddSiLUD", typeof(GgmlStorage))]
        public static Tensor AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG) => ExecuteActivationGrad(result, srcG, srcW, resG, GgmlActivationGradOp.SiLU, "AddSiLUD");

        [RegisterOpStorageType("layernorm", typeof(GgmlStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps)
            => ExecuteNorm(result, src, gamma, beta, eps, GgmlNormOp.LayerNorm, "layernorm");

        [RegisterOpStorageType("rmsnorm", typeof(GgmlStorage))]
        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps)
            => ExecuteNorm(result, src, gamma, beta, eps, GgmlNormOp.RmsNorm, "rmsnorm");

        [RegisterOpStorageType("layernormgrad", typeof(GgmlStorage))]
        public static Tensor LayerNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps)
            => ExecuteNormGrad(result, gradGamma, gradBeta, adj, y, x, gamma, beta, eps, GgmlNormOp.LayerNorm, "layernormgrad");

        [RegisterOpStorageType("rmsnormgrad", typeof(GgmlStorage))]
        public static Tensor RMSNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps)
            => ExecuteNormGrad(result, gradGamma, gradBeta, adj, y, x, gamma, beta, eps, GgmlNormOp.RmsNorm, "rmsnormgrad");

        [RegisterOpStorageType("indexselect", typeof(GgmlStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            ValidateIndexSelectArguments(result, src, indice, isAdd);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, new long[] { indice.Sizes[0], src.Sizes[1] });
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView2D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView2D srcView)
                || !TryCreateContiguousTensor(indice, out GgmlContiguousTensor indexTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML indexselect requires Float32 row-contiguous source/result matrices and a contiguous Float32/Int32 index vector.");
            }

            GgmlNative.IndexSelect(resultView, srcView, indexTensor, isAdd);
            return writeTarget;
        }

        [RegisterOpStorageType("indexselectgrad", typeof(GgmlStorage))]
        public static Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            ValidateIndexSelectGradArguments(grad, adj, indice);

            if (!TryCreateStandardView(grad, out GgmlTensorView2D gradView)
                || !TryCreateStandardView(adj, out GgmlTensorView2D adjView)
                || !TryCreateContiguousTensor(indice, out GgmlContiguousTensor indexTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML indexselectgrad requires Float32 row-contiguous gradient/adjoint matrices and a contiguous Float32/Int32 index vector.");
            }

            GgmlNative.IndexSelectGrad(gradView, adjView, indexTensor);
            return grad;
        }

        [RegisterOpStorageType("repeat_interleave", typeof(GgmlStorage))]
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (dim < 0 || dim >= src.DimensionCount)
                throw new ArgumentOutOfRangeException(nameof(dim));
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            long[] resultSizes = src.Sizes.ToArray();
            resultSizes[dim] *= repeats;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, resultSizes);

            using var contiguousSrc = Ops.AsContiguous(src);

            int sliceCount = 1;
            for (int d = 0; d < dim; d++)
                sliceCount *= (int)contiguousSrc.Sizes[d];

            int innerSize = 1;
            for (int d = dim + 1; d < contiguousSrc.DimensionCount; d++)
                innerSize *= (int)contiguousSrc.Sizes[d];

            int dimSize = (int)contiguousSrc.Sizes[dim];
            int sliceSize = innerSize;

            unsafe
            {
                float* srcPtr = (float*)GetBufferStart(contiguousSrc);
                float* dstPtr = (float*)GetBufferStart(writeTarget);

                for (int outer = 0; outer < sliceCount; outer++)
                {
                    float* srcBatch = srcPtr + outer * dimSize * sliceSize;
                    float* dstBatch = dstPtr + outer * dimSize * repeats * sliceSize;
                    for (int i = 0; i < dimSize; i++)
                    {
                        float* srcSlice = srcBatch + i * sliceSize;
                        for (int r = 0; r < repeats; r++)
                        {
                            float* dstSlice = dstBatch + (i * repeats + r) * sliceSize;
                            System.Buffer.MemoryCopy(srcSlice, dstSlice, sliceSize * sizeof(float), sliceSize * sizeof(float));
                        }
                    }
                }
            }

            return writeTarget;
        }

        [RegisterOpStorageType("add_causal_mask", typeof(GgmlStorage))]
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (!tensor.IsContiguous())
                throw new InvalidOperationException("AddCausalMask requires a contiguous tensor.");

            int ndim = tensor.DimensionCount;
            long cols = tensor.Sizes[ndim - 1];
            long totalElements = tensor.ElementCount();
            long totalRows = totalElements / cols;

            unsafe
            {
                float* ptr = (float*)GetBufferStart(tensor);
                for (int row = 0; row < totalRows; row++)
                {
                    int t = row % seqLen;
                    int threshold = startPos + t;
                    int sStart = Math.Max(0, threshold + 1);
                    float* rowPtr = ptr + row * cols;
                    for (int s = sStart; s < cols; s++)
                    {
                        rowPtr[s] += maskedValue;
                    }
                }
            }
        }

        [RegisterOpStorageType("rope", typeof(GgmlStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            ValidateRoPEArguments(result, src, seqLen, "rope");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException("GGML rope requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.RoPE(resultView, srcView, seqLen, rowOffset);
            return writeTarget;
        }

        [RegisterOpStorageType("ropegrad", typeof(GgmlStorage))]
        public static Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen, int rowOffset)
        {
            ValidateRoPEArguments(grad, adj, seqLen, "ropegrad");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad, adj, false, adj.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView))
            {
                throw new NotSupportedException("GGML ropegrad requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.RoPEGrad(resultView, adjView, seqLen, rowOffset);
            return writeTarget;
        }

        [RegisterOpStorageType("rope_ex", typeof(GgmlStorage))]
        public static Tensor RoPEEx(
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
            bool addToResult = false,
            bool invertPositions = false)
        {
            ValidateRoPEExArguments(result, src, positions, ropeDim, "rope_ex");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateContiguousTensor(positions, out GgmlContiguousTensor positionTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML rope_ex requires Float32 source/result tensors and contiguous Float32/Int32 positions.");
            }

            GgmlNative.RoPEEx(
                resultView,
                srcView,
                positionTensor,
                ropeDim,
                mode,
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow,
                addToResult,
                invertPositions);

            return writeTarget;
        }

        // GGML-only helper: same as RoPEEx but additionally applies a
        // per-dim freq_factors scaling (used by Gemma 4 global layers'
        // proportional RoPE). `freqFactors` must be a contiguous Float32
        // tensor of length ropeDim/2 (or null to skip scaling).
        public static Tensor RoPEExWithFreqFactors(
            Tensor result,
            Tensor src,
            Tensor positions,
            Tensor freqFactors,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult = false,
            bool invertPositions = false)
        {
            ValidateRoPEExArguments(result, src, positions, ropeDim, "rope_ex_ff");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateContiguousTensor(positions, out GgmlContiguousTensor positionTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML rope_ex_ff requires Float32 source/result tensors and contiguous Float32/Int32 positions.");
            }

            IntPtr freqFactorsPtr = IntPtr.Zero;
            int freqFactorsLen = 0;
            if (freqFactors != null)
            {
                if (freqFactors.ElementType != DType.Float32)
                    throw new NotSupportedException("GGML rope_ex_ff freq_factors must be Float32.");
                freqFactorsPtr = GetBufferStart(freqFactors);
                freqFactorsLen = (int)freqFactors.ElementCount();
            }

            GgmlNative.RoPEExWithFreqFactors(
                resultView,
                srcView,
                positionTensor,
                ropeDim,
                mode,
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow,
                addToResult,
                invertPositions,
                freqFactorsPtr,
                freqFactorsLen);

            return writeTarget;
        }

        // GGML-only helper: interleaved/blocked multi-axis RoPE (ggml_rope_multi
        // / kernel_rope_multi). Input shape must be [headDim, numHeads, seqLen, 1]
        // (4-D, batch=1). Positions must be a contiguous I32 vector of length
        // 4*seqLen, with per-axis-concatenated layout
        //   pos[0..n-1]   = T  positions
        //   pos[n..2n-1]  = H  positions
        //   pos[2n..3n-1] = W  positions
        //   pos[3n..4n-1] = 4th axis (zeros for static images)
        // `sections` is the four mrope_section entries (Qwen3.5 ships
        // [11,11,10,0]). `mode` should include GGML_ROPE_TYPE_MROPE (8) for
        // blocked-section MRoPE or GGML_ROPE_TYPE_IMROPE (40) for interleaved.
        public static Tensor RoPEMRoPE(
            Tensor result,
            Tensor src,
            Tensor positions,
            int[] sections,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor = 0f,
            float attnFactor = 1f,
            float betaFast = 0f,
            float betaSlow = 0f)
        {
            if (sections == null || sections.Length != 4)
                throw new ArgumentException("sections must be a length-4 int[] (Qwen3.5 ships [11,11,10,0]).", nameof(sections));
            // Custom validation: ggml_rope_multi expects 4 positions per token
            // (T, H, W, +1 extra axis), so the positions tensor length is 4*seqLen
            // rather than the seqLen*numHeads rope_ex convention.
            ValidateGgmlTensor(src, nameof(src), "rope_mrope");
            ValidateGgmlIndexTensor(positions, nameof(positions), "rope_mrope");
            if (src.DimensionCount != 4)
                throw new NotSupportedException("rope_mrope requires a 4D input tensor [batch=1, seqLen, numHeads, headDim].");
            if (ropeDim <= 0 || ropeDim > src.Sizes[^1] || (ropeDim & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(ropeDim), "rope_mrope requires an even ropeDim within the last tensor dimension.");
            long seqLen = src.Sizes[1]; // C# View(batch, seqLen, numHeads, headDim) → Sizes[1] = seqLen
            if (positions.ElementCount() != 4 * seqLen)
                throw new InvalidOperationException(
                    $"rope_mrope expects positions length 4*seqLen ({4 * seqLen}); got {positions.ElementCount()}.");
            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "rope_mrope");
                if (!HasSameShape(result, src))
                    throw new InvalidOperationException("rope_mrope expects result and src to have the same shape.");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateContiguousTensor(positions, out GgmlContiguousTensor positionTensor, DType.Float32, DType.Int32))
            {
                throw new NotSupportedException("GGML rope_mrope requires Float32 source/result tensors and contiguous Float32/Int32 positions.");
            }

            GgmlNative.RoPEMRoPE(
                resultView,
                srcView,
                positionTensor,
                ropeDim,
                mode,
                sections[0], sections[1], sections[2], sections[3],
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow);

            return writeTarget;
        }

        [RegisterOpStorageType("float2half", typeof(GgmlStorage))]
        public Tensor Float2Half(Tensor result, Tensor src)
        {
            throw new NotSupportedException("GGML backends currently support Float32 tensors only. Disable AMP to use this backend.");
        }

        [RegisterOpStorageType("half2float", typeof(GgmlStorage))]
        public Tensor Half2Float(Tensor result, Tensor src)
        {
            throw new NotSupportedException("GGML backends currently support Float32 tensors only. Disable AMP to use this backend.");
        }

        private static Tensor ExecuteUnary(Tensor result, Tensor src, GgmlUnaryOp op, string opName)
        {
            ValidateUnaryArguments(result, src, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Unary(op, resultView, srcView);
            return writeTarget;
        }

        private static Tensor ExecuteFusedActMul(Tensor result, Tensor a, Tensor b, GgmlFusedActMulOp op, string opName)
        {
            ValidateBinaryTensorArguments(result, a, b, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, a, false, a.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(a, out GgmlTensorView4D aView)
                || !TryCreateStandardView(b, out GgmlTensorView4D bView))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.FusedActMul(op, resultView, aView, bView);
            return writeTarget;
        }

        private static Tensor ExecuteFusedActMulSplit(Tensor result, Tensor gateUp, int halfDim, GgmlFusedActMulOp op, string opName)
        {
            if (gateUp == null)
                throw new ArgumentNullException(nameof(gateUp));
            if (halfDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(halfDim));
            if (gateUp.ElementType != DType.Float32)
                throw new InvalidOperationException($"GGML {opName} requires a Float32 gate_up tensor.");
            if (gateUp.DimensionCount != 2 || gateUp.Sizes[1] != 2L * halfDim)
                throw new InvalidOperationException($"GGML {opName} requires gate_up shape [N, 2*halfDim].");

            long rows = gateUp.Sizes[0];
            long[] resultShape = new long[] { rows, halfDim };

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp, false, resultShape);
            if (!TryCreateRawView(writeTarget, out GgmlTensorView2D resultView)
                || !TryCreateRawView(gateUp, out GgmlTensorView2D gateUpView))
            {
                throw new NotSupportedException($"GGML {opName} requires native-backed Float32 2D tensors.");
            }
            if (resultView.Stride1 != 1 || gateUpView.Stride1 != 1)
            {
                throw new NotSupportedException($"GGML {opName} requires row-major (stride1==1) tensors.");
            }
            if (gateUpView.Stride0 < gateUpView.Dim1)
            {
                throw new NotSupportedException($"GGML {opName} requires gate_up.stride0 >= gate_up.dim1.");
            }

            GgmlNative.FusedActMulSplit(op, resultView, gateUpView, halfDim);
            return writeTarget;
        }

        private static Tensor ExecuteBinaryTensor(Tensor result, Tensor lhs, Tensor rhs, GgmlBinaryTensorOp op, string opName)
        {
            ValidateBinaryTensorArguments(result, lhs, rhs, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            Tensor compactRhs = null;
            try
            {
                if (TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                    && TryCreateStandardView(lhs, out GgmlTensorView4D lhsView)
                    && TryCreateStandardOrBroadcastView(rhs, out GgmlTensorView4D rhsView, out compactRhs))
                {
                    GgmlNative.BinaryTensor(op, resultView, lhsView, rhsView);
                    return writeTarget;
                }

                if (op == GgmlBinaryTensorOp.Add
                    && HasSameShape(writeTarget, rhs)
                    && AreEquivalentViews(writeTarget, lhs)
                    && HasExpandedWriteDimension(writeTarget))
                {
                    return ExecuteAtomicAddHost(writeTarget, rhs);
                }

                if (!CanUseHostElementwiseWrite(writeTarget))
                {
                    throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous or broadcast-expand compatible layout.");
                }

                return ExecuteBinaryTensorHost(writeTarget, lhs, rhs, op, opName);
            }
            finally
            {
                compactRhs?.Dispose();
            }
        }

        private static Tensor ExecuteBinaryScalar(Tensor result, Tensor src, float scalar, GgmlBinaryScalarOp op, string opName)
        {
            ValidateBinaryScalarArguments(result, src, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.BinaryScalar(op, resultView, srcView, scalar);
            return writeTarget;
        }

        private static unsafe Tensor ExecuteBinaryTensorHost(Tensor result, Tensor lhs, Tensor rhs, GgmlBinaryTensorOp op, string opName)
        {
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.SizesMemory, result.StridesMemory);
            TensorIterState lhsIter = new TensorIterState((float*)GetBufferStart(lhs), lhs.DimensionCount, lhs.SizesMemory, lhs.StridesMemory);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.SizesMemory, rhs.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !lhsIter.ReachedBlockEnd() && !rhsIter.ReachedBlockEnd();
                    resultIter.BlockStep(), lhsIter.BlockStep(), rhsIter.BlockStep())
                {
                    float lhsValue = *lhsIter.data;
                    float rhsValue = *rhsIter.data;
                    *resultIter.data = ApplyBinaryTensorOp(lhsValue, rhsValue, op, opName);
                }
            } while (resultIter.NextBlock() && lhsIter.NextBlock() && rhsIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("atomicadd", typeof(GgmlStorage))]
        public static Tensor AtomicAdd(Tensor result, Tensor rhs)
        {
            ValidateBinaryTensorArguments(result, result, rhs, "atomicadd");
            return ExecuteAtomicAddHost(result, rhs);
        }

        private static unsafe Tensor ExecuteAtomicAddHost(Tensor result, Tensor rhs)
        {
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.SizesMemory, result.StridesMemory);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.SizesMemory, rhs.StridesMemory);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !rhsIter.ReachedBlockEnd(); resultIter.BlockStep(), rhsIter.BlockStep())
                {
                    *resultIter.data += *rhsIter.data;
                }
            } while (resultIter.NextBlock() && rhsIter.NextBlock());

            return result;
        }

        private static float ApplyBinaryTensorOp(float lhsValue, float rhsValue, GgmlBinaryTensorOp op, string opName)
        {
            return op switch
            {
                GgmlBinaryTensorOp.Add => lhsValue + rhsValue,
                GgmlBinaryTensorOp.Sub => lhsValue - rhsValue,
                GgmlBinaryTensorOp.Mul => lhsValue * rhsValue,
                GgmlBinaryTensorOp.Div => lhsValue / rhsValue,
                _ => throw new NotSupportedException($"Host fallback does not support GGML binary tensor op '{opName}'."),
            };
        }

        private static Tensor ExecuteActivationGrad(Tensor result, Tensor accumulation, Tensor src, Tensor grad, GgmlActivationGradOp op, string opName)
        {
            ValidateActivationGradArguments(result, accumulation, src, grad, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView4D accumulationView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateStandardView(grad, out GgmlTensorView4D gradView)
                || (accumulation != null && !TryCreateStandardView(accumulation, out accumulationView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.ActivationGrad(op, resultView, srcView, gradView, accumulationView, accumulation != null);
            return writeTarget;
        }

        private static Tensor ExecuteNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps, GgmlNormOp op, string opName)
        {
            ValidateNormArguments(result, src, gamma, beta, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            GgmlTensorView4D betaView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateStandardView(gamma, out GgmlTensorView4D gammaView)
                || (beta != null && !TryCreateStandardView(beta, out betaView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Norm(op, resultView, srcView, gammaView, betaView, beta != null, eps);
            return writeTarget;
        }

        private static Tensor ExecuteNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps, GgmlNormOp op, string opName)
        {
            ValidateNormGradArguments(result, gradGamma, gradBeta, adj, y, x, gamma, beta, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, adj, false, adj.Sizes);
            GgmlTensorView4D gradBetaView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(gradGamma, out GgmlTensorView4D gradGammaView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView)
                || !TryCreateStandardView(x, out GgmlTensorView4D xView)
                || !TryCreateStandardView(gamma, out GgmlTensorView4D gammaView)
                || (gradBeta != null && !TryCreateStandardView(gradBeta, out gradBetaView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.NormGrad(op, resultView, gradGammaView, gradBetaView, adjView, xView, gammaView, gradBeta != null, eps);
            return writeTarget;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView2D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor) || !TryCreateRawView(tensor, out GgmlTensorView2D rawView))
            {
                view = default;
                return false;
            }

            view = rawView;
            return true;
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView2D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView3D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor) || !TryCreateRawView(tensor, out GgmlTensorView3D rawView))
            {
                view = default;
                return false;
            }

            view = rawView;
            return true;
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView3D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView4D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor)
                || tensor.DimensionCount > 4
                || !HasNativeBufferStorage(tensor)
                || !TryGetRawSpanBytes(tensor, out long rawBytes))
            {
                view = default;
                return false;
            }

            long[] paddedSizes = new long[] { 1, 1, 1, 1 };
            long[] paddedStrides = new long[] { 0, 0, 0, 0 };
            int offset = 4 - tensor.DimensionCount;

            for (int i = 0; i < tensor.DimensionCount; ++i)
            {
                paddedSizes[offset + i] = tensor.Sizes[i];
                paddedStrides[offset + i] = tensor.Strides[i];
            }

            try
            {
                for (int i = 2; i >= 0; --i)
                {
                    long requiredStride = checked(paddedStrides[i + 1] * paddedSizes[i + 1]);
                    if (paddedSizes[i] == 1 || i < offset)
                    {
                        paddedStrides[i] = requiredStride;
                    }
                }

                if (!TryGetInt32(paddedSizes[3], out int ne0)
                    || !TryGetInt32(paddedSizes[2], out int ne1)
                    || !TryGetInt32(paddedSizes[1], out int ne2)
                    || !TryGetInt32(paddedSizes[0], out int ne3))
                {
                    view = default;
                    return false;
                }

                long elementSize = tensor.ElementType.Size();
                long nb1 = checked(paddedStrides[2] * elementSize);
                long nb2 = checked(paddedStrides[1] * elementSize);
                long nb3 = checked(paddedStrides[0] * elementSize);

                view = new GgmlTensorView4D(GetBufferStart(tensor), ne0, ne1, ne2, ne3, nb1, nb2, nb3, rawBytes);
                return true;
            }
            catch (OverflowException)
            {
                view = default;
                return false;
            }
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView4D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateContiguousTensor(Tensor tensor, out GgmlContiguousTensor contiguousTensor, params DType[] allowedTypes)
        {
            if (!HasNativeBufferStorage(tensor) || !tensor.IsContiguous())
            {
                contiguousTensor = default;
                return false;
            }

            if (allowedTypes != null && allowedTypes.Length > 0 && Array.IndexOf(allowedTypes, tensor.ElementType) < 0)
            {
                contiguousTensor = default;
                return false;
            }

            contiguousTensor = new GgmlContiguousTensor(GetBufferStart(tensor), tensor.ElementCount(), tensor.ElementType);
            return true;
        }

        private static bool TryCreateRawView(Tensor tensor, out GgmlTensorView2D view)
        {
            if (tensor.DimensionCount != 2
                || tensor.ElementType != DType.Float32
                || !HasNativeBufferStorage(tensor)
                || !TryGetRawSpanBytes(tensor, out long rawBytes)
                || !TryGetInt32(tensor.Sizes[0], out int dim0)
                || !TryGetInt32(tensor.Sizes[1], out int dim1)
                || !TryGetInt32(tensor.Strides[0], out int stride0)
                || !TryGetInt32(tensor.Strides[1], out int stride1))
            {
                view = default;
                return false;
            }

            view = new GgmlTensorView2D(GetBufferStart(tensor), dim0, dim1, stride0, stride1, rawBytes);
            return true;
        }

        private static bool TryCreateRawView(Tensor tensor, out GgmlTensorView3D view)
        {
            if (tensor.DimensionCount != 3
                || tensor.ElementType != DType.Float32
                || !HasNativeBufferStorage(tensor)
                || !TryGetRawSpanBytes(tensor, out long rawBytes)
                || !TryGetInt32(tensor.Sizes[0], out int dim0)
                || !TryGetInt32(tensor.Sizes[1], out int dim1)
                || !TryGetInt32(tensor.Sizes[2], out int dim2)
                || !TryGetInt32(tensor.Strides[0], out int stride0)
                || !TryGetInt32(tensor.Strides[1], out int stride1)
                || !TryGetInt32(tensor.Strides[2], out int stride2))
            {
                view = default;
                return false;
            }

            view = new GgmlTensorView3D(GetBufferStart(tensor), dim0, dim1, dim2, stride0, stride1, stride2, rawBytes);
            return true;
        }

        private static bool CanMapTensorToStandardGgmlView(Tensor tensor)
        {
            if (tensor.ElementType != DType.Float32
                || tensor.DimensionCount < 1
                || tensor.DimensionCount > 4
                || tensor.Strides[tensor.DimensionCount - 1] != 1)
            {
                return false;
            }

            long requiredStride = 1;
            for (int dimension = tensor.DimensionCount - 1; dimension >= 0; --dimension)
            {
                long size = tensor.Sizes[dimension];
                if (size <= 0)
                {
                    return false;
                }

                if (size == 1)
                {
                    continue;
                }

                long stride = tensor.Strides[dimension];
                if (stride < requiredStride)
                {
                    return false;
                }

                try
                {
                    requiredStride = checked(stride * size);
                }
                catch (OverflowException)
                {
                    return false;
                }
            }

            return true;
        }

        private static bool TryCreateCompactedBroadcastTensor(Tensor tensor, out Tensor compactTensor)
        {
            compactTensor = null;

            if (tensor.ElementType != DType.Float32
                || !HasNativeBufferStorage(tensor)
                || tensor.DimensionCount < 1
                || tensor.DimensionCount > 4
                || tensor.Strides[tensor.DimensionCount - 1] != 1)
            {
                return false;
            }

            long[] compactSizes = tensor.Sizes.ToArray();
            long[] compactStrides = tensor.Strides.ToArray();
            bool changed = false;

            for (int i = 0; i < compactSizes.Length; ++i)
            {
                if (compactSizes[i] <= 0)
                {
                    return false;
                }

                if (compactSizes[i] > 1 && compactStrides[i] == 0)
                {
                    compactSizes[i] = 1;
                    changed = true;
                }
            }

            if (!changed)
            {
                return false;
            }

            compactTensor = new Tensor(compactSizes, compactStrides, tensor.Storage, tensor.StorageOffset);
            if (!CanMapTensorToStandardGgmlView(compactTensor))
            {
                compactTensor.Dispose();
                compactTensor = null;
                return false;
            }

            return true;
        }

        private static void ValidateMaskResultTensor(Tensor result, string opName)
        {
            ValidateGgmlTensor(result, nameof(result), opName);

            if (result.DimensionCount < 1)
            {
                throw new InvalidOperationException($"{opName} requires a tensor with at least one dimension.");
            }

            if (!result.IsContiguous())
            {
                throw new NotSupportedException($"{opName} currently requires a contiguous result tensor.");
            }
        }

        private static void ValidateMaskLengthsTensor(Tensor tensor, string argumentName, string opName)
        {
            ValidateGgmlTensor(tensor, argumentName, opName);

            if (!tensor.IsContiguous())
            {
                throw new NotSupportedException($"{opName} currently requires contiguous length tensors.");
            }
        }

        private static void ValidateGatherArguments(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateGgmlTensor(src, nameof(src), "gather");
            ValidateGgmlIndexTensor(indices, nameof(indices), "gather");

            if (dim < 0 || dim >= src.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("gather expects src and indices to have the same number of dimensions.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "gather");
                if (result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException("gather expects result and src to have the same number of dimensions.");
                }

                if (!result.IsSameSizeAs(indices))
                {
                    throw new InvalidOperationException("gather expects result and indices to have the same shape.");
                }

                if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
                {
                    throw new InvalidOperationException("gather expects result and src to match in every dimension except dim.");
                }
            }
        }

        private static void ValidateScatterArguments(Tensor result, Tensor src, int dim, Tensor indices, string opName)
        {
            ValidateGgmlTensor(result, nameof(result), opName);
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlIndexTensor(indices, nameof(indices), opName);

            if (dim < 0 || dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (result.DimensionCount != src.DimensionCount || indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException($"{opName} expects result, src, and indices to have the same number of dimensions.");
            }

            if (!src.IsSameSizeAs(indices))
            {
                throw new InvalidOperationException($"{opName} expects src and indices to have the same shape.");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException($"{opName} expects result and src to match in every dimension except dim.");
            }
        }

        private static void ValidateScatterFillArguments(Tensor result, int dim, Tensor indices)
        {
            ValidateGgmlTensor(result, nameof(result), "scatter_fill");
            ValidateGgmlIndexTensor(indices, nameof(indices), "scatter_fill");

            if (dim < 0 || dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (indices.DimensionCount != result.DimensionCount)
            {
                throw new InvalidOperationException("scatter_fill expects result and indices to have the same number of dimensions.");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(indices.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("scatter_fill expects result and indices to match in every dimension except dim.");
            }
        }

        private static void ValidateElementwiseArguments(Tensor result, string opName, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException($"{opName} requires at least one tensor argument.", nameof(tensors));
            }

            Tensor reference = tensors[0];
            ValidateGgmlTensor(reference, "tensor0", opName);

            for (int i = 1; i < tensors.Length; ++i)
            {
                ValidateGgmlTensor(tensors[i], $"tensor{i}", opName);
                if (!HasSameShape(reference, tensors[i]))
                {
                    throw new InvalidOperationException($"{opName} expects all tensor arguments to have the same shape.");
                }
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, reference))
                {
                    throw new InvalidOperationException($"{opName} expects result to have the same shape as its tensor inputs.");
                }
            }
        }

        private static void GetFlatRowsCols(Tensor tensor, string opName, out int rows, out int cols)
        {
            long colsLong = tensor.Sizes[tensor.DimensionCount - 1];
            long storageSize = TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides);
            if (colsLong <= 0 || (storageSize % colsLong) != 0)
            {
                throw new InvalidOperationException($"{opName} received an invalid tensor layout.");
            }

            long rowsLong = storageSize / colsLong;
            if (!TryGetInt32(rowsLong, out rows) || !TryGetInt32(colsLong, out cols))
            {
                throw new NotSupportedException($"{opName} tensor dimensions exceed the GGML mask builder limits.");
            }
        }

        private static bool TryGetRawSpanBytes(Tensor tensor, out long rawBytes)
        {
            try
            {
                long maxOffset = 0;
                for (int dimension = 0; dimension < tensor.DimensionCount; ++dimension)
                {
                    long size = tensor.Sizes[dimension];
                    if (size <= 0)
                    {
                        rawBytes = 0;
                        return false;
                    }

                    maxOffset = checked(maxOffset + checked((size - 1) * tensor.Strides[dimension]));
                }

                rawBytes = checked((maxOffset + 1) * tensor.ElementType.Size());
                return true;
            }
            catch (OverflowException)
            {
                rawBytes = 0;
                return false;
            }
        }

        private static bool CanUseHostElementwiseWrite(Tensor tensor)
        {
            for (int dimension = 0; dimension < tensor.DimensionCount; ++dimension)
            {
                if (tensor.Sizes[dimension] > 1 && tensor.Strides[dimension] == 0)
                {
                    return false;
                }
            }

            return true;
        }

        private static bool TryGetInt32(long value, out int intValue)
        {
            try
            {
                intValue = checked((int)value);
                return true;
            }
            catch (OverflowException)
            {
                intValue = 0;
                return false;
            }
        }

        private static bool HasNativeBufferStorage(Tensor tensor) =>
            tensor != null && (tensor.Storage is GgmlStorage || tensor.Storage is CpuStorage);

        private static IntPtr GetBufferStart(Tensor tensor)
        {
            return tensor.Storage switch
            {
                GgmlStorage ggmlStorage => ggmlStorage.PtrAtElement(tensor.StorageOffset),
                CpuStorage cpuStorage => cpuStorage.PtrAtElement(tensor.StorageOffset),
                _ => throw new ArgumentException("Tensor storage must expose a native CPU-accessible buffer.", nameof(tensor)),
            };
        }

        private static void ValidateAddmmArguments(Tensor result, Tensor src, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type. src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result?.ElementType}'");
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }

            if (!(m1.Storage is GgmlStorage))
            {
                throw new ArgumentException("m1 must be a GGML tensor", nameof(m1));
            }

            if (!(m2.Storage is GgmlStorage))
            {
                throw new ArgumentException("m2 must be a GGML tensor", nameof(m2));
            }

            if (src.DimensionCount != 2 || m1.DimensionCount != 2 || m2.DimensionCount != 2)
            {
                throw new ArgumentException("addmm expects 2D tensors.");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
            {
                throw new InvalidOperationException("Size mismatch");
            }
        }

        private static void ValidateAddmmBatchArguments(Tensor result, Tensor src, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type. src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result?.ElementType}'");
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }

            if (!(m1.Storage is GgmlStorage))
            {
                throw new ArgumentException("m1 must be a GGML tensor", nameof(m1));
            }

            if (!(m2.Storage is GgmlStorage))
            {
                throw new ArgumentException("m2 must be a GGML tensor", nameof(m2));
            }

            if (src.DimensionCount != 3 || m1.DimensionCount != 3 || m2.DimensionCount != 3)
            {
                throw new ArgumentException("addmmbatch expects 3D tensors.");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[0] != m2.Sizes[0] || src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException("Size mismatch");
            }
        }

        private static void ValidateSoftmaxArguments(Tensor result, Tensor src)
        {
            if (src.ElementType != DType.Float32 || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"softmax expects Float32 tensors. src = '{src.ElementType}', result = '{result?.ElementType}'");
            }

            if (!(src.Storage is GgmlStorage))
            {
                throw new ArgumentException("src must be a GGML tensor", nameof(src));
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }
        }

        private static void ValidateSoftmaxGradArguments(Tensor grad, Tensor adj, Tensor val)
        {
            if (adj.ElementType != DType.Float32 || val.ElementType != DType.Float32 || (grad != null && grad.ElementType != DType.Float32))
            {
                throw new InvalidOperationException($"softmaxgrad expects Float32 tensors. grad = '{grad?.ElementType}', adj = '{adj.ElementType}', val = '{val.ElementType}'");
            }

            if (!(adj.Storage is GgmlStorage))
            {
                throw new ArgumentException("adj must be a GGML tensor", nameof(adj));
            }

            if (!(val.Storage is GgmlStorage))
            {
                throw new ArgumentException("val must be a GGML tensor", nameof(val));
            }

            if (grad != null && !(grad.Storage is GgmlStorage))
            {
                throw new ArgumentException("grad must be a GGML tensor", nameof(grad));
            }

            if (adj.DimensionCount != val.DimensionCount)
            {
                throw new InvalidOperationException("adj and val must have the same number of dimensions.");
            }

            if (!HasSameShape(adj, val))
            {
                throw new InvalidOperationException("adj and val must have the same shape.");
            }

            if (grad != null && !HasSameShape(grad, adj))
            {
                throw new InvalidOperationException("grad and adj must have the same shape.");
            }
        }

        private static void ValidateAdamArguments(Tensor tw, Tensor tg, Tensor tv, Tensor tm)
        {
            if (tw.ElementType != DType.Float32 || tg.ElementType != DType.Float32 || tv.ElementType != DType.Float32 || tm.ElementType != DType.Float32)
            {
                throw new InvalidOperationException($"adam expects Float32 tensors. weight = '{tw.ElementType}', gradient = '{tg.ElementType}', v = '{tv.ElementType}', m = '{tm.ElementType}'");
            }

            if (!(tw.Storage is GgmlStorage))
            {
                throw new ArgumentException("weight must be a GGML tensor", nameof(tw));
            }

            if (!(tg.Storage is GgmlStorage))
            {
                throw new ArgumentException("gradient must be a GGML tensor", nameof(tg));
            }

            if (!(tv.Storage is GgmlStorage))
            {
                throw new ArgumentException("v must be a GGML tensor", nameof(tv));
            }

            if (!(tm.Storage is GgmlStorage))
            {
                throw new ArgumentException("m must be a GGML tensor", nameof(tm));
            }

            if (!HasSameShape(tw, tg)
                || !HasSameShape(tw, tv)
                || !HasSameShape(tw, tm))
            {
                throw new InvalidOperationException("weight, gradient, v, and m must have the same shape.");
            }
        }

        private static void ValidateCopyArguments(Tensor result, Tensor src)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }
            if (src == null)
            {
                throw new ArgumentNullException(nameof(src));
            }
            if (!(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }
            if (!(src.Storage is GgmlStorage))
            {
                throw new ArgumentException("src must be a GGML tensor", nameof(src));
            }

            // Copy is dtype-preserving: KV-cache resize and similar callers narrow
            // the same storage on both sides, so cross-dtype copy here would just
            // mask an upstream bug. Float32/Float16/Q8_0 are the storage classes
            // models actually instantiate (see KvCacheStorage), so accept those.
            if (result.ElementType != src.ElementType)
            {
                throw new InvalidOperationException(
                    $"copy expects source and result to share an element type (result={result.ElementType}, src={src.ElementType}).");
            }
            if (result.ElementType != DType.Float32
                && result.ElementType != DType.Float16
                && result.ElementType != DType.Q8_0)
            {
                throw new InvalidOperationException(
                    $"copy supports Float32, Float16, or Q8_0 tensors (got {result.ElementType}).");
            }

            if (result.ElementCount() != src.ElementCount())
            {
                throw new InvalidOperationException("copy expects source and result to have the same number of elements.");
            }
        }

        private static unsafe Tensor ExecuteReduction(Tensor result, Tensor src, int dimension, bool divideByCount, string opName)
        {
            ValidateReductionArguments(result, src, dimension, opName);

            long[] desiredSize = src.Sizes.ToArray();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeReduction(writeTarget, src, dimension, divideByCount ? GgmlReductionOp.Mean : GgmlReductionOp.Sum))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.SizesMemory, src.StridesMemory, dimension);

            do
            {
                float sum = 0.0f;
                for (long i = 0; i < srcIter.size; ++i)
                {
                    sum += *(srcIter.data + i * srcIter.stride);
                }

                *resultIter.data = divideByCount ? (sum / srcIter.size) : sum;
            } while (resultIter.NextBlock() && srcIter.NextBlock());

            return writeTarget;
        }

        private static unsafe Tensor ExecuteIndexReduction(Tensor result, Tensor src, int dimension, bool selectMinimum, string opName)
        {
            ValidateReductionArguments(result, src, dimension, opName);

            long[] desiredSize = src.Sizes.ToArray();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeIndexReduction(writeTarget, src, dimension, selectMinimum ? GgmlIndexReductionOp.Argmin : GgmlIndexReductionOp.Argmax))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.SizesMemory, writeTarget.StridesMemory, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.SizesMemory, src.StridesMemory, dimension);

            do
            {
                long bestIndex = 0;
                float bestValue = *srcIter.data;
                for (long i = 1; i < srcIter.size; ++i)
                {
                    float currentValue = *(srcIter.data + i * srcIter.stride);
                    if (selectMinimum ? currentValue < bestValue : currentValue > bestValue)
                    {
                        bestValue = currentValue;
                        bestIndex = i;
                    }
                }

                *resultIter.data = bestIndex;
            } while (resultIter.NextBlock() && srcIter.NextBlock());

            return writeTarget;
        }

        private static bool TryExecuteNativeReduction(Tensor result, Tensor src, int dimension, GgmlReductionOp op)
        {
            if (dimension != src.DimensionCount - 1)
            {
                return false;
            }

            if (!TryCreateStandardView(result, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                return false;
            }

            GgmlNative.ReduceLastDim(op, resultView, srcView);
            return true;
        }

        private static bool TryExecuteNativeIndexReduction(Tensor result, Tensor src, int dimension, GgmlIndexReductionOp op)
        {
            if (dimension != src.DimensionCount - 1)
            {
                return false;
            }

            if (!TryCreateStandardView(result, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                return false;
            }

            GgmlNative.IndexReduction(op, resultView, srcView);
            return true;
        }

        private static void ValidateReductionArguments(Tensor result, Tensor src, int dimension, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (dimension < 0 || dimension >= src.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            if (src.Sizes[dimension] <= 0)
            {
                throw new InvalidOperationException($"{opName} expects a non-empty reduction dimension.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);

                if (result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same number of dimensions.");
                }

                for (int i = 0; i < src.DimensionCount; ++i)
                {
                    long expected = i == dimension ? 1 : src.Sizes[i];
                    if (result.Sizes[i] != expected)
                    {
                        throw new InvalidOperationException($"{opName} expects result to match src except for a singleton reduction dimension.");
                    }
                }
            }
        }

        private static void ValidateUnaryArguments(Tensor result, Tensor src, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateBinaryTensorArguments(Tensor result, Tensor lhs, Tensor rhs, string opName)
        {
            ValidateGgmlTensor(lhs, nameof(lhs), opName);
            ValidateGgmlTensor(rhs, nameof(rhs), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, lhs))
                {
                    throw new InvalidOperationException($"{opName} expects result and lhs to have the same shape.");
                }
            }
        }

        private static void ValidateBinaryScalarArguments(Tensor result, Tensor src, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateActivationGradArguments(Tensor result, Tensor accumulation, Tensor src, Tensor grad, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlTensor(grad, nameof(grad), opName);

            if (!HasSameShape(src, grad))
            {
                throw new InvalidOperationException($"{opName} expects src and grad to have the same shape.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }

            if (accumulation != null)
            {
                ValidateGgmlTensor(accumulation, nameof(accumulation), opName);
                if (!HasSameShape(accumulation, src))
                {
                    throw new InvalidOperationException($"{opName} expects accumulation and src to have the same shape.");
                }
            }
        }

        private static void ValidateNormArguments(Tensor result, Tensor src, Tensor gamma, Tensor beta, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlTensor(gamma, nameof(gamma), opName);

            if (src.DimensionCount < 2 || src.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if (gamma.ElementCount() != src.Sizes[src.DimensionCount - 1])
            {
                throw new InvalidOperationException($"{opName} expects gamma element count to match the last dimension of src.");
            }

            if (beta != null)
            {
                ValidateGgmlTensor(beta, nameof(beta), opName);
                if (beta.ElementCount() != src.Sizes[src.DimensionCount - 1])
                {
                    throw new InvalidOperationException($"{opName} expects beta element count to match the last dimension of src.");
                }
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateNormGradArguments(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, string opName)
        {
            ValidateGgmlTensor(adj, nameof(adj), opName);
            ValidateGgmlTensor(y, nameof(y), opName);
            ValidateGgmlTensor(x, nameof(x), opName);
            ValidateGgmlTensor(gamma, nameof(gamma), opName);
            ValidateGgmlTensor(gradGamma, nameof(gradGamma), opName);

            if (x.DimensionCount < 2 || x.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if (!HasSameShape(adj, x) || !HasSameShape(y, x))
            {
                throw new InvalidOperationException($"{opName} expects adj, y, and x to have the same shape.");
            }

            if (gamma.ElementCount() != x.Sizes[x.DimensionCount - 1])
            {
                throw new InvalidOperationException($"{opName} expects gamma element count to match the last dimension of x.");
            }

            if (!HasSameShape(gradGamma, gamma))
            {
                throw new InvalidOperationException($"{opName} expects gradGamma to have the same shape as gamma.");
            }

            if (beta != null)
            {
                ValidateGgmlTensor(beta, nameof(beta), opName);
                if (beta.ElementCount() != x.Sizes[x.DimensionCount - 1])
                {
                    throw new InvalidOperationException($"{opName} expects beta element count to match the last dimension of x.");
                }

                if (gradBeta == null)
                {
                    throw new ArgumentNullException(nameof(gradBeta), $"{opName} requires gradBeta when beta is provided.");
                }

                ValidateGgmlTensor(gradBeta, nameof(gradBeta), opName);
                if (!HasSameShape(gradBeta, beta))
                {
                    throw new InvalidOperationException($"{opName} expects gradBeta to have the same shape as beta.");
                }
            }
            else if (gradBeta != null)
            {
                throw new InvalidOperationException($"{opName} does not accept gradBeta when beta is null.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, x))
                {
                    throw new InvalidOperationException($"{opName} expects result and x to have the same shape.");
                }
            }
        }

        private static void ValidateIndexSelectArguments(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            ValidateGgmlTensor(src, nameof(src), "indexselect");
            ValidateGgmlIndexTensor(indice, nameof(indice), "indexselect");

            if (isAdd && result == null)
            {
                throw new ArgumentNullException(nameof(result), "indexselect with isAdd=true requires an existing result tensor.");
            }

            if (src.DimensionCount != 2)
            {
                throw new NotSupportedException("GGML indexselect currently supports 2D source tensors only.");
            }

            if (!IsSupportedIndexTensor(indice))
            {
                throw new NotSupportedException("GGML indexselect currently supports contiguous 1D or Nx1 index tensors only.");
            }

            if (!indice.IsContiguous())
            {
                throw new NotSupportedException("GGML indexselect requires contiguous indices.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "indexselect");
                if (result.DimensionCount != 2 || result.Sizes[0] != indice.Sizes[0] || result.Sizes[1] != src.Sizes[1])
                {
                    throw new InvalidOperationException("indexselect expects result shape [indices, src_cols].");
                }
            }
        }

        private static void ValidateMulMatIdArguments(Tensor result, Tensor expertWeights, Tensor input, Tensor ids)
        {
            ValidateGgmlTensor(expertWeights, nameof(expertWeights), "mulmatid");
            ValidateGgmlTensor(input, nameof(input), "mulmatid");
            ValidateGgmlIndexTensor(ids, nameof(ids), "mulmatid");

            if (expertWeights.DimensionCount != 3 || input.DimensionCount != 3)
            {
                throw new NotSupportedException("mulmatid expects 3D expertWeights and input tensors.");
            }

            if (ids.DimensionCount != 2)
            {
                throw new NotSupportedException("mulmatid expects a 2D id tensor.");
            }

            if (!ids.IsContiguous())
            {
                throw new NotSupportedException("mulmatid expects contiguous ids.");
            }

            if (expertWeights.Sizes[2] != input.Sizes[2])
            {
                throw new InvalidOperationException("mulmatid expects expertWeights/input to match on the inner dimension.");
            }

            if (ids.Sizes[0] != input.Sizes[0] || ids.Sizes[1] % input.Sizes[1] != 0)
            {
                throw new InvalidOperationException("mulmatid expects ids shape [tokens, expert_used] aligned with input shape [tokens, expert_used_or_broadcast, cols].");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "mulmatid");
                long[] expected = new long[] { input.Sizes[0], ids.Sizes[1], expertWeights.Sizes[1] };
                if (!TensorResultBuilder.ArrayEqual(result.Sizes, expected))
                {
                    throw new InvalidOperationException("mulmatid expects result shape [tokens, expert_used, rows].");
                }
            }
        }

        private static void ValidateScaledDotProductAttentionArguments(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask)
        {
            ValidateGgmlTensor(query, nameof(query), "scaled_dot_product_attention");
            ValidateGgmlTensor(key, nameof(key), "scaled_dot_product_attention");
            ValidateGgmlTensor(value, nameof(value), "scaled_dot_product_attention");

            if (query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4)
            {
                throw new NotSupportedException("scaled_dot_product_attention expects query/key/value to be 4D tensors.");
            }

            if (query.Sizes[0] != key.Sizes[0] || query.Sizes[0] != value.Sizes[0])
            {
                throw new InvalidOperationException("scaled_dot_product_attention expects query/key/value to share the batch dimension.");
            }

            if (query.Sizes[2] != key.Sizes[2] || query.Sizes[2] != value.Sizes[2])
            {
                throw new InvalidOperationException("scaled_dot_product_attention expects query/key/value to share the head dimension.");
            }

            if (query.Sizes[3] != key.Sizes[3])
            {
                throw new InvalidOperationException("scaled_dot_product_attention expects query/key to share the key depth.");
            }

            if (mask != null)
            {
                ValidateGgmlTensor(mask, nameof(mask), "scaled_dot_product_attention");
                if (mask.DimensionCount != 4)
                {
                    throw new NotSupportedException("scaled_dot_product_attention expects a 4D mask tensor.");
                }

                if (mask.Sizes[3] != key.Sizes[1] || mask.Sizes[2] != query.Sizes[1] || mask.Sizes[1] != query.Sizes[2])
                {
                    throw new InvalidOperationException("scaled_dot_product_attention expects mask shape [batch, heads, seq_q, seq_k].");
                }
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "scaled_dot_product_attention");
                long[] expected = new long[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
                if (!TensorResultBuilder.ArrayEqual(result.Sizes, expected))
                {
                    throw new InvalidOperationException("scaled_dot_product_attention expects result shape [batch, seq_q, heads, value_dim].");
                }
            }
        }

        private static void ValidateAddIdArguments(Tensor result, Tensor src, Tensor bias, Tensor ids)
        {
            ValidateGgmlTensor(src, nameof(src), "addid");
            ValidateGgmlTensor(bias, nameof(bias), "addid");
            ValidateGgmlIndexTensor(ids, nameof(ids), "addid");

            if (src.DimensionCount != 3)
            {
                throw new NotSupportedException("addid expects a 3D source tensor.");
            }

            if (bias.DimensionCount != 2 || ids.DimensionCount != 2)
            {
                throw new NotSupportedException("addid expects 2D bias and id tensors.");
            }

            if (!ids.IsContiguous())
            {
                throw new NotSupportedException("addid expects contiguous ids.");
            }

            if (src.Sizes[2] != bias.Sizes[1] || src.Sizes[0] != ids.Sizes[0] || src.Sizes[1] != ids.Sizes[1])
            {
                throw new InvalidOperationException("addid expects src shape [tokens, expert_used, rows], bias shape [experts, rows], and ids shape [tokens, expert_used].");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "addid");
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException("addid expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateIndexSelectGradArguments(Tensor grad, Tensor adj, Tensor indice)
        {
            ValidateGgmlTensor(grad, nameof(grad), "indexselectgrad");
            ValidateGgmlTensor(adj, nameof(adj), "indexselectgrad");
            ValidateGgmlIndexTensor(indice, nameof(indice), "indexselectgrad");

            if (grad.DimensionCount != 2 || adj.DimensionCount != 2)
            {
                throw new NotSupportedException("GGML indexselectgrad currently supports 2D gradient and adjoint tensors only.");
            }

            if (!IsSupportedIndexTensor(indice))
            {
                throw new NotSupportedException("GGML indexselectgrad currently supports contiguous 1D or Nx1 index tensors only.");
            }

            if (!indice.IsContiguous())
            {
                throw new NotSupportedException("GGML indexselectgrad requires contiguous indices.");
            }

            if (adj.Sizes[0] != indice.Sizes[0] || grad.Sizes[1] != adj.Sizes[1])
            {
                throw new InvalidOperationException("indexselectgrad expects adj shape [indices, grad_cols].");
            }
        }

        private static void ValidateRoPEArguments(Tensor result, Tensor src, int seqLen, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (seqLen <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen), $"{opName} requires seqLen > 0.");
            }

            if (src.DimensionCount < 2 || src.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if ((src.Sizes[src.DimensionCount - 1] & 1) != 0)
            {
                throw new NotSupportedException($"{opName} requires the last tensor dimension to be even.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateRoPEExArguments(Tensor result, Tensor src, Tensor positions, int ropeDim, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlIndexTensor(positions, nameof(positions), opName);

            if (src.DimensionCount < 2 || src.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if (ropeDim <= 0 || ropeDim > src.Sizes[src.DimensionCount - 1] || (ropeDim & 1) != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(ropeDim), $"{opName} requires an even ropeDim within the last tensor dimension.");
            }

            long rows = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides) / src.Sizes[^1];
            if (positions.ElementCount() != rows)
            {
                throw new InvalidOperationException($"{opName} expects one explicit position value per logical row.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateGgmlTensor(Tensor tensor, string argumentName, string opName)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(argumentName);
            }

            if (tensor.ElementType != DType.Float32)
            {
                throw new InvalidOperationException($"{opName} expects Float32 tensors only.");
            }

            if (!(tensor.Storage is GgmlStorage))
            {
                throw new ArgumentException($"{argumentName} must be a GGML tensor", argumentName);
            }
        }

        private static void ValidateGgmlIndexTensor(Tensor tensor, string argumentName, string opName)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(argumentName);
            }

            if (tensor.ElementType != DType.Float32 && tensor.ElementType != DType.Int32)
            {
                throw new InvalidOperationException($"{opName} expects Float32 or Int32 index tensors.");
            }

            if (!(tensor.Storage is GgmlStorage))
            {
                throw new ArgumentException($"{argumentName} must be a GGML tensor", argumentName);
            }
        }

        private static unsafe long ReadIndexValue(TensorDimIterState iter, DType elementType, int logicalIndex)
        {
            return elementType switch
            {
                DType.Int32 => *(((int*)iter.data) + logicalIndex * iter.stride),
                _ => (long)*(((float*)iter.data) + logicalIndex * iter.stride),
            };
        }

        private static bool HasSameShape(Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount != rhs.DimensionCount)
            {
                return false;
            }

            for (int i = 0; i < lhs.DimensionCount; ++i)
            {
                if (lhs.Sizes[i] != rhs.Sizes[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static bool AreEquivalentViews(Tensor lhs, Tensor rhs)
        {
            if (!ReferenceEquals(lhs.Storage, rhs.Storage) || lhs.StorageOffset != rhs.StorageOffset)
            {
                return false;
            }

            if (lhs.DimensionCount != rhs.DimensionCount)
            {
                return false;
            }

            for (int i = 0; i < lhs.DimensionCount; ++i)
            {
                if (lhs.Sizes[i] != rhs.Sizes[i] || lhs.Strides[i] != rhs.Strides[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static bool HasExpandedWriteDimension(Tensor tensor)
        {
            for (int i = 0; i < tensor.DimensionCount; ++i)
            {
                if (tensor.Sizes[i] > 1 && tensor.Strides[i] == 0)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsSupportedIndexTensor(Tensor indice)
        {
            if (indice.DimensionCount == 1)
            {
                return true;
            }

            if (indice.DimensionCount == 2 && indice.Sizes[1] == 1)
            {
                return true;
            }

            return false;
        }
    }
}
