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

namespace TensorSharp.GGML
{
    public static class GgmlLossOps
    {
        public static bool CanUseNativeCrossEntropyLoss(Tensor probs, Tensor targetIndices)
        {
            if (probs == null || targetIndices == null)
            {
                return false;
            }

            if (probs.ElementType != DType.Float32
                || (targetIndices.ElementType != DType.Float32 && targetIndices.ElementType != DType.Int32)
                || probs.Storage is not GgmlStorage
                || targetIndices.Storage is not GgmlStorage
                || probs.DimensionCount < 1
                || probs.DimensionCount > 4
                || !probs.IsContiguous()
                || !targetIndices.IsContiguous())
            {
                return false;
            }

            long classCount = probs.Sizes[^1];
            if (classCount <= 0)
            {
                return false;
            }

            return targetIndices.ElementCount() == (probs.ElementCount() / classCount);
        }

        public static float CrossEntropyLoss(Tensor probs, Tensor targetIndices, float smooth = 0.0f, float labelSmooth = 0.0f)
        {
            ValidateLossInputs(probs, targetIndices, "crossentropyloss");
            return GgmlNative.CrossEntropyLoss(CreateStandardView(probs), CreateContiguousTensor(targetIndices), smooth, labelSmooth);
        }

        public static void CrossEntropyLossBackward(Tensor grad, Tensor probs, Tensor targetIndices, float lossGradient, float smooth = 0.0f, float labelSmooth = 0.0f, bool addGrad = true)
        {
            ValidateLossInputs(probs, targetIndices, "crossentropyloss_backward");

            if (grad == null)
            {
                throw new ArgumentNullException(nameof(grad));
            }

            if (grad.ElementType != DType.Float32 || grad.Storage is not GgmlStorage || !grad.IsContiguous())
            {
                throw new NotSupportedException("crossentropyloss_backward requires a contiguous Float32 GGML gradient tensor.");
            }

            if (!grad.IsSameSizeAs(probs))
            {
                throw new InvalidOperationException("crossentropyloss_backward requires gradient and probability tensors to have the same shape.");
            }

            GgmlNative.CrossEntropyLossBackward(CreateStandardView(grad), CreateStandardView(probs), CreateContiguousTensor(targetIndices), lossGradient, smooth, labelSmooth, addGrad);
        }

        private static void ValidateLossInputs(Tensor probs, Tensor targetIndices, string opName)
        {
            if (!CanUseNativeCrossEntropyLoss(probs, targetIndices))
            {
                throw new NotSupportedException($"{opName} requires contiguous GGML tensors, Float32 probabilities, Int32/Float32 target indices, 1 to 4 probability dimensions, and one target index per logical row.");
            }
        }

        private static GgmlTensorView4D CreateStandardView(Tensor tensor)
        {
            long[] paddedSizes = new long[] { 1, 1, 1, 1 };
            long[] paddedStrides = new long[] { 0, 0, 0, 0 };
            int offset = 4 - tensor.DimensionCount;

            for (int i = 0; i < tensor.DimensionCount; ++i)
            {
                paddedSizes[offset + i] = tensor.Sizes[i];
                paddedStrides[offset + i] = tensor.Strides[i];
            }

            for (int i = 2; i >= 0; --i)
            {
                long requiredStride = checked(paddedStrides[i + 1] * paddedSizes[i + 1]);
                if (paddedSizes[i] == 1 || i < offset)
                {
                    paddedStrides[i] = requiredStride;
                }
            }

            long elementSize = tensor.ElementType.Size();
            long rawBytes = checked(tensor.ElementCount() * elementSize);

            return new GgmlTensorView4D(
                GetBufferStart(tensor),
                checked((int)paddedSizes[3]),
                checked((int)paddedSizes[2]),
                checked((int)paddedSizes[1]),
                checked((int)paddedSizes[0]),
                checked(paddedStrides[2] * elementSize),
                checked(paddedStrides[1] * elementSize),
                checked(paddedStrides[0] * elementSize),
                rawBytes);
        }

        private static GgmlContiguousTensor CreateContiguousTensor(Tensor tensor)
        {
            return new GgmlContiguousTensor(GetBufferStart(tensor), tensor.ElementCount(), tensor.ElementType);
        }

        private static IntPtr GetBufferStart(Tensor tensor)
        {
            return ((GgmlStorage)tensor.Storage).PtrAtElement(tensor.StorageOffset);
        }
    }
}
