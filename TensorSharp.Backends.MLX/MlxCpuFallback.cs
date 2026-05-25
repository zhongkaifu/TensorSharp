using System;
using System.Collections.Generic;
using TensorSharp.Cpu;
using TensorSharp.Core;

namespace TensorSharp.MLX
{
    internal enum MlxFallbackReturnKind
    {
        Void,
        Tensor,
        Raw
    }

    internal static class MlxCpuFallback
    {
        private static readonly CpuAllocator CpuAllocator = new(BlasEnum.DotNet);

        public static object Invoke(string opName, MlxFallbackReturnKind returnKind, int[] modifiedTensorIndexes, object[] args)
        {
            if (string.Equals(opName, "SiLUMulSplit", StringComparison.Ordinal))
                return SiLUMulSplit((Tensor)args[0], (Tensor)args[1], (int)args[2]);

            if (string.Equals(opName, "scaled_dot_product_attention", StringComparison.Ordinal))
                return ScaledDotProductAttention((Tensor)args[0], (Tensor)args[1], (Tensor)args[2], (Tensor)args[3], (Tensor)args[4], (float)args[5]);

            object returnValue = InvokeCpu(opName, args, out Dictionary<Tensor, Tensor> mappedTensors);
            try
            {
                foreach (int modifiedTensorIndex in modifiedTensorIndexes)
                {
                    if (modifiedTensorIndex >= 0 &&
                        modifiedTensorIndex < args.Length &&
                        args[modifiedTensorIndex] is Tensor modifiedTensor &&
                        mappedTensors.TryGetValue(modifiedTensor, out Tensor cpuTensor))
                    {
                        CopyLogical(modifiedTensor, cpuTensor);
                    }
                }

                if (returnKind == MlxFallbackReturnKind.Raw)
                    return returnValue;

                if (returnKind == MlxFallbackReturnKind.Void)
                    return null;

                if (args.Length > 0 && args[0] is Tensor explicitResult)
                    return explicitResult;

                if (returnValue is Tensor cpuReturn)
                {
                    Tensor mlxReturn = CreateMlxLike(cpuReturn, args);
                    CopyLogical(mlxReturn, cpuReturn);
                    return mlxReturn;
                }

                return null;
            }
            finally
            {
                DisposeMapped(mappedTensors, returnValue as Tensor);
            }
        }

        private static object InvokeCpu(string opName, object[] args, out Dictionary<Tensor, Tensor> mappedTensors)
        {
            mappedTensors = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);
            object[] cpuArgs = new object[args.Length];

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] is Tensor tensor)
                {
                    if (!mappedTensors.TryGetValue(tensor, out Tensor cpuTensor))
                    {
                        cpuTensor = ToCpuTensor(tensor);
                        mappedTensors.Add(tensor, cpuTensor);
                    }

                    cpuArgs[i] = cpuTensor;
                }
                else
                {
                    cpuArgs[i] = args[i];
                }
            }

            return OpRegistry.Invoke(opName, cpuArgs);
        }

        private static Tensor ToCpuTensor(Tensor source)
        {
            Tensor cpu = new(CpuAllocator, source.ElementType, source.Sizes);
            CopyLogical(cpu, source);
            return cpu;
        }

        private static Tensor CreateMlxLike(Tensor source, object[] originalArgs)
        {
            IAllocator allocator = null;
            foreach (object arg in originalArgs)
            {
                if (arg is Tensor tensor && tensor.Storage is MlxStorage)
                {
                    allocator = tensor.Allocator;
                    break;
                }
            }

            allocator ??= new MlxAllocator();
            return new Tensor(allocator, source.ElementType, source.Sizes);
        }

        internal static void CopyLogical(Tensor destination, Tensor source)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (destination.ElementType != source.ElementType)
                throw new InvalidOperationException("Source and destination tensors must have the same element type.");
            if (destination.ElementCount() != source.ElementCount())
                throw new InvalidOperationException("Source and destination tensors must have the same number of elements.");
            if (destination.DimensionCount != source.DimensionCount)
                throw new InvalidOperationException("Source and destination tensors must have the same rank.");

            for (int i = 0; i < source.DimensionCount; i++)
            {
                if (destination.Sizes[i] != source.Sizes[i])
                    throw new InvalidOperationException("Source and destination tensors must have the same shape.");
            }

            if (source.DimensionCount == 0)
            {
                CopyElement(destination, destination.StorageOffset, source, source.StorageOffset);
                return;
            }

            CopyRecursive(destination, source, 0, destination.StorageOffset, source.StorageOffset);
        }

        private static void CopyRecursive(Tensor destination, Tensor source, int dimension, long destinationOffset, long sourceOffset)
        {
            if (dimension == source.DimensionCount)
            {
                CopyElement(destination, destinationOffset, source, sourceOffset);
                return;
            }

            long size = source.Sizes[dimension];
            long sourceStride = source.Strides[dimension];
            long destinationStride = destination.Strides[dimension];
            for (long i = 0; i < size; i++)
            {
                CopyRecursive(
                    destination,
                    source,
                    dimension + 1,
                    destinationOffset + i * destinationStride,
                    sourceOffset + i * sourceStride);
            }
        }

        private static unsafe void CopyElement(Tensor destination, long destinationOffset, Tensor source, long sourceOffset)
        {
            long byteCount = source.ElementType.Size();
            byte* tmp = stackalloc byte[(int)byteCount];
            source.Storage.CopyFromStorage((IntPtr)tmp, sourceOffset, byteCount);
            destination.Storage.CopyToStorage(destinationOffset, (IntPtr)tmp, byteCount);
        }

        private static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            if (gateUp == null)
                throw new ArgumentNullException(nameof(gateUp));
            if (halfDim <= 0 || gateUp.DimensionCount != 2 || gateUp.Sizes[1] < halfDim * 2L)
                throw new ArgumentException("SiLUMulSplit expects a [tokens, 2*halfDim] tensor.", nameof(gateUp));

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            using Tensor gate = gateUp.Narrow(1, 0, halfDim);
            using Tensor up = gateUp.Narrow(1, halfDim, halfDim);
            Ops.Copy(writeTarget, gate);
            Ops.SiLUMul(writeTarget, writeTarget, up);
            return writeTarget;
        }

        private static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (key == null)
                throw new ArgumentNullException(nameof(key));
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (query.ElementType != DType.Float32 || key.ElementType != DType.Float32 || value.ElementType != DType.Float32)
                throw new NotSupportedException("MLX fallback scaled-dot-product attention currently supports Float32 tensors only.");
            if (query.DimensionCount != 4 || key.DimensionCount != 4 || value.DimensionCount != 4)
                throw new ArgumentException("Scaled-dot-product attention expects query/key/value tensors with shape [batch, heads, seq, dim].");
            if (query.Sizes[0] != key.Sizes[0] || query.Sizes[0] != value.Sizes[0] ||
                query.Sizes[1] != key.Sizes[1] || query.Sizes[1] != value.Sizes[1] ||
                key.Sizes[2] != value.Sizes[2] || query.Sizes[3] != key.Sizes[3])
            {
                throw new InvalidOperationException("Scaled-dot-product attention tensor shapes are incompatible.");
            }

            long batch = query.Sizes[0];
            long heads = query.Sizes[1];
            long queryLen = query.Sizes[2];
            long keyLen = key.Sizes[2];
            long headDim = query.Sizes[3];
            long valueDim = value.Sizes[3];
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, DType.Float32, false, batch, heads, queryLen, valueDim);

            if (keyLen > int.MaxValue)
                throw new NotSupportedException("MLX fallback attention does not support key lengths above Int32.MaxValue.");

            float[] scores = new float[(int)keyLen];
            for (long b = 0; b < batch; b++)
            {
                for (long h = 0; h < heads; h++)
                {
                    for (long q = 0; q < queryLen; q++)
                    {
                        float max = float.NegativeInfinity;
                        for (long k = 0; k < keyLen; k++)
                        {
                            float dot = 0.0f;
                            for (long d = 0; d < headDim; d++)
                                dot += query.GetElementAsFloat(b, h, q, d) * key.GetElementAsFloat(b, h, k, d);

                            float score = dot * scale + ReadAttentionMask(mask, b, h, q, k);
                            scores[k] = score;
                            if (score > max)
                                max = score;
                        }

                        float sum = 0.0f;
                        for (long k = 0; k < keyLen; k++)
                        {
                            float exp = MathF.Exp(scores[k] - max);
                            scores[k] = exp;
                            sum += exp;
                        }

                        float invSum = sum == 0.0f ? 0.0f : 1.0f / sum;
                        for (long d = 0; d < valueDim; d++)
                        {
                            float acc = 0.0f;
                            for (long k = 0; k < keyLen; k++)
                                acc += scores[k] * invSum * value.GetElementAsFloat(b, h, k, d);
                            writeTarget.SetElementAsFloat(acc, b, h, q, d);
                        }
                    }
                }
            }

            return writeTarget;
        }

        private static float ReadAttentionMask(Tensor mask, long b, long h, long q, long k)
        {
            if (mask == null)
                return 0.0f;
            if (mask.ElementType != DType.Float32)
                throw new NotSupportedException("Attention mask must be Float32.");

            return mask.DimensionCount switch
            {
                2 => mask.GetElementAsFloat(q, k),
                3 => mask.GetElementAsFloat(b, q, k),
                4 => mask.GetElementAsFloat(b, h, q, k),
                _ => throw new NotSupportedException("Attention mask must be rank 2, 3, or 4."),
            };
        }

        private static void DisposeMapped(Dictionary<Tensor, Tensor> mappedTensors, Tensor returnedTensor)
        {
            var disposed = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
            foreach (Tensor tensor in mappedTensors.Values)
            {
                if (disposed.Add(tensor))
                    tensor.Dispose();
            }

            if (returnedTensor != null && disposed.Add(returnedTensor))
                returnedTensor.Dispose();
        }
    }
}
