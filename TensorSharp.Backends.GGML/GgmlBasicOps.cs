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
using TensorSharp.Core;
using TensorSharp.Cpu;

namespace TensorSharp.GGML
{
    [OpsClass]
    public class GgmlBasicOps
    {
        [RegisterOpStorageType("fill", typeof(GgmlStorage))]
        public static unsafe void Fill(Tensor result, float value)
        {
            ValidateGgmlTensor(result, nameof(result), "fill");

            float* buffer = (float*)GetBufferStart(result);
            TensorIterState iter = new TensorIterState(buffer, result.DimensionCount, result.Sizes, result.Strides);

            do
            {
                for (; !iter.ReachedBlockEnd(); iter.BlockStep())
                {
                    *iter.data = value;
                }
            } while (iter.NextBlock());
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

        [RegisterOpStorageType("gather", typeof(GgmlStorage))]
        public static unsafe Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateGatherArguments(result, src, dim, indices);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);
            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

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

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

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

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

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

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

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

        public static IntPtr AlignedAlloc(long size) => GgmlNative.AlignedAlloc(size);
        public static void AlignedFree(IntPtr ptr) => GgmlNative.AlignedFree(ptr);
        public static void ClearHostBufferCache() => GgmlNative.ClearHostBufferCache();
        public static void InvalidateHostBuffer(IntPtr ptr) => GgmlNative.InvalidateHostBuffer(ptr);
        public static void SyncHostBuffer(IntPtr ptr, long byteCount) => GgmlNative.SyncHostBuffer(ptr, byteCount);
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
            int intermediateSize, int ropeMode)
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
                intermediateSize, ropeMode);
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
            int intermediateSize, int ropeMode)
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
                intermediateSize, ropeMode);
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
                kCache.ElementType != DType.Float32 || vCache.ElementType != DType.Float32 || output.ElementType != DType.Float32)
                throw new ArgumentException("Flash attention decode requires F32 tensors.");

            IntPtr qPtr = GetBufferStart(q);
            IntPtr kPtr = GetBufferStart(k);
            IntPtr vPtr = GetBufferStart(v);
            IntPtr kcPtr = GetBufferStart(kCache);
            IntPtr vcPtr = GetBufferStart(vCache);
            IntPtr outPtr = GetBufferStart(output);

            GgmlNative.FlashAttnDecode(
                qPtr, kPtr, vPtr,
                kcPtr, vcPtr, outPtr,
                numHeads, numKvHeads, headDim,
                maxSeqLen, position, scale);
        }

        /// <summary>
        /// Single-token Qwen3.5 full-attention decode kernel. Performs the entire FullAttention
        /// block (RMSNorm, fused QKV, deinterleave Q/gate, per-head QK norm, RoPE, KV cache append,
        /// flash attention, sigmoid-gated mix, output projection + residual add) in a single GGML
        /// graph dispatch. Reduces ~2 standalone GGML calls + ~6 small CPU/GPU sync points per
        /// attention layer per decode token to one fused dispatch.
        /// </summary>
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
                qNorm.ElementType != DType.Float32 || kNorm.ElementType != DType.Float32 ||
                kCache.ElementType != DType.Float32 || vCache.ElementType != DType.Float32)
                throw new ArgumentException("Qwen3.5 attention layer decode requires F32 tensors.");
            if (qkvData == IntPtr.Zero || oData == IntPtr.Zero)
                throw new ArgumentException("Qwen3.5 attention layer decode requires non-zero quantized weight pointers.");

            IntPtr residualPtr = GetBufferStart(residual);
            IntPtr attnNormPtr = GetBufferStart(attnNorm);
            IntPtr qNormPtr = GetBufferStart(qNorm);
            IntPtr kNormPtr = GetBufferStart(kNorm);
            IntPtr kCachePtr = GetBufferStart(kCache);
            IntPtr vCachePtr = GetBufferStart(vCache);
            int hiddenSize = (int)residual.Sizes[residual.Sizes.Length - 1];

            GgmlNative.Qwen35AttentionLayerDecode(
                residualPtr, hiddenSize,
                attnNormPtr,
                qkvData, qkvGgmlType, qkvNe0, qkvNe1, qkvRawBytes,
                qNormPtr, kNormPtr, headDim,
                oData, oGgmlType, oNe0, oNe1, oRawBytes,
                kCachePtr, vCachePtr,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale, ropeMode);
        }

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
            IntPtr[] plePostNormArr)
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
                plePostNormArr);
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

            float* resultBuffer = (float*)GetBufferStart(result);
            float* srcBuffer = (float*)GetBufferStart(src);

            if (result.IsContiguous() && src.IsContiguous())
            {
                long byteCount = checked(elementCount * result.ElementType.Size());
                Buffer.MemoryCopy(srcBuffer, resultBuffer, byteCount, byteCount);
                return;
            }

            TensorIterState resultIter = new TensorIterState(resultBuffer, result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState srcIter = new TensorIterState(srcBuffer, src.DimensionCount, src.Sizes, src.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !srcIter.ReachedBlockEnd(); resultIter.BlockStep(), srcIter.BlockStep())
                {
                    *resultIter.data = *srcIter.data;
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock());
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
            TensorIterState iter = new TensorIterState(buffer, src.DimensionCount, src.Sizes, src.Strides);
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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);

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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);

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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);

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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);
            TensorIterState wIter = new TensorIterState((float*)GetBufferStart(w), w.DimensionCount, w.Sizes, w.Strides);

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

            long[] resultSizes = (long[])src.Sizes.Clone();
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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState lhsIter = new TensorIterState((float*)GetBufferStart(lhs), lhs.DimensionCount, lhs.Sizes, lhs.Strides);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.Sizes, rhs.Strides);

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
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.Sizes, rhs.Strides);

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

            long[] compactSizes = (long[])tensor.Sizes.Clone();
            long[] compactStrides = (long[])tensor.Strides.Clone();
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

            ValidateGgmlTensor(result, nameof(result), "copy");
            ValidateGgmlTensor(src, nameof(src), "copy");

            if (result.ElementCount() != src.ElementCount())
            {
                throw new InvalidOperationException("copy expects source and result to have the same number of elements.");
            }
        }

        private static unsafe Tensor ExecuteReduction(Tensor result, Tensor src, int dimension, bool divideByCount, string opName)
        {
            ValidateReductionArguments(result, src, dimension, opName);

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeReduction(writeTarget, src, dimension, divideByCount ? GgmlReductionOp.Mean : GgmlReductionOp.Sum))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dimension);

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

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeIndexReduction(writeTarget, src, dimension, selectMinimum ? GgmlIndexReductionOp.Argmin : GgmlIndexReductionOp.Argmax))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dimension);

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
