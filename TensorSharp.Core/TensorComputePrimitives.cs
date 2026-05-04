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
using System.Numerics;
using System.Runtime.CompilerServices;

namespace TensorSharp
{
    public static unsafe class TensorComputePrimitives
    {
        public static IntPtr GetStoragePointer(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            return tensor.Storage.PtrAtElement(tensor.StorageOffset);
        }

        public static IntPtr GetStorageBasePointer(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            return tensor.Storage.PtrAtElement(0);
        }

        // GetFloatPointer / GetHalfPointer hand out raw host pointers that callers
        // dereference on the CPU. Under async-compute on the GGML/Metal backend,
        // the bytes behind those pointers may still be receiving writes from a
        // GPU command buffer that hasn't completed yet, so we drain any pending
        // work before returning the pointer. The drain is cheap when nothing's
        // pending (single atomic check on the C++ side).
        //
        // Native op binding code uses tensor.Storage.PtrAtElement directly (see
        // GgmlBasicOps.GetBufferStart) and intentionally bypasses this hook so
        // that GPU-only chaining stays asynchronous.
        public static float* GetFloatPointer(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.ElementType != DType.Float32)
                throw new NotSupportedException($"Requires a Float32 tensor, but found {tensor.ElementType}");

            tensor.Storage.EnsureHostReadable();
            return (float*)GetStoragePointer(tensor);
        }

        public static ushort* GetHalfPointer(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.ElementType != DType.Float16)
                throw new NotSupportedException($"Requires a Float16 tensor, but found {tensor.ElementType}");

            tensor.Storage.EnsureHostReadable();
            return (ushort*)GetStoragePointer(tensor);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> LoadVector(float* p) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StoreVector(float* p, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref *(byte*)p, v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot(float* a, float* b, int n)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen * 2; i += vLen * 2)
            {
                acc0 += LoadVector(a + i) * LoadVector(b + i);
                acc1 += LoadVector(a + i + vLen) * LoadVector(b + i + vLen);
            }
            var acc = acc0 + acc1;
            for (; i <= n - vLen; i += vLen)
                acc += LoadVector(a + i) * LoadVector(b + i);
            float sum = Vector.Sum(acc);
            for (; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SumSquares(float* a, int n)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen * 2; i += vLen * 2)
            {
                var v0 = LoadVector(a + i);
                var v1 = LoadVector(a + i + vLen);
                acc0 += v0 * v0;
                acc1 += v1 * v1;
            }
            var acc = acc0 + acc1;
            for (; i <= n - vLen; i += vLen)
            {
                var v = LoadVector(a + i);
                acc += v * v;
            }
            float sum = Vector.Sum(acc);
            for (; i < n; i++)
                sum += a[i] * a[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(float* data, float scale, int n)
        {
            int vLen = Vector<float>.Count;
            var vs = new Vector<float>(scale);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StoreVector(data + i, LoadVector(data + i) * vs);
            for (; i < n; i++)
                data[i] *= scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScaleAdd(float* dst, float* src, float weight, int n)
        {
            int vLen = Vector<float>.Count;
            var vw = new Vector<float>(weight);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StoreVector(dst + i, LoadVector(dst + i) + LoadVector(src + i) * vw);
            for (; i < n; i++)
                dst[i] += weight * src[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dot4(float* a0, float* a1, float* a2, float* a3,
            float* b, int n,
            out float r0, out float r1, out float r2, out float r3)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            var acc2 = Vector<float>.Zero;
            var acc3 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var vb = LoadVector(b + i);
                acc0 += LoadVector(a0 + i) * vb;
                acc1 += LoadVector(a1 + i) * vb;
                acc2 += LoadVector(a2 + i) * vb;
                acc3 += LoadVector(a3 + i) * vb;
            }
            float s0 = Vector.Sum(acc0);
            float s1 = Vector.Sum(acc1);
            float s2 = Vector.Sum(acc2);
            float s3 = Vector.Sum(acc3);
            for (; i < n; i++)
            {
                float bi = b[i];
                s0 += a0[i] * bi;
                s1 += a1[i] * bi;
                s2 += a2[i] * bi;
                s3 += a3[i] * bi;
            }
            r0 = s0;
            r1 = s1;
            r2 = s2;
            r3 = s3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScaleAdd4(float* d0, float* d1, float* d2, float* d3,
            float* src, float w0, float w1, float w2, float w3, int n)
        {
            int vLen = Vector<float>.Count;
            var vw0 = new Vector<float>(w0);
            var vw1 = new Vector<float>(w1);
            var vw2 = new Vector<float>(w2);
            var vw3 = new Vector<float>(w3);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var vs = LoadVector(src + i);
                StoreVector(d0 + i, LoadVector(d0 + i) + vs * vw0);
                StoreVector(d1 + i, LoadVector(d1 + i) + vs * vw1);
                StoreVector(d2 + i, LoadVector(d2 + i) + vs * vw2);
                StoreVector(d3 + i, LoadVector(d3 + i) + vs * vw3);
            }
            for (; i < n; i++)
            {
                float s = src[i];
                d0[i] += w0 * s;
                d1[i] += w1 * s;
                d2[i] += w2 * s;
                d3[i] += w3 * s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubScale(float* dst, float* a, float* b, float scale, int n)
        {
            int vLen = Vector<float>.Count;
            var vs = new Vector<float>(scale);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StoreVector(dst + i, (LoadVector(a + i) - LoadVector(b + i)) * vs);
            for (; i < n; i++)
                dst[i] = (a[i] - b[i]) * scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Zero(float* data, int n)
        {
            int vLen = Vector<float>.Count;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StoreVector(data + i, Vector<float>.Zero);
            for (; i < n; i++)
                data[i] = 0;
        }

        // ====================================================================
        // Float16 helpers for the quantized KV cache. F16 storage is read into
        // F32 accumulators on the fly: on Apple Silicon and modern x86 the
        // F16->F32 conversion is essentially free in registers, so the dot
        // product runs at twice the effective memory bandwidth of an F32
        // cache when the cache is big enough to miss the LLC.
        // ====================================================================

        /// <summary>
        /// Convert a contiguous block of <paramref name="n"/> float values to
        /// IEEE 754 binary16 (half) and write them into <paramref name="dst"/>.
        /// Uses <see cref="System.Half"/> which the BCL implements with the
        /// hardware FP16 instructions when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void F32ToF16(ushort* dst, float* src, int n)
        {
            for (int i = 0; i < n; i++)
                dst[i] = BitConverter.HalfToUInt16Bits((System.Half)src[i]);
        }

        /// <summary>
        /// Convert a contiguous block of <paramref name="n"/> half-precision
        /// values to F32. Useful when a downstream kernel only consumes F32.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void F16ToF32(float* dst, ushort* src, int n)
        {
            for (int i = 0; i < n; i++)
                dst[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
        }

        /// <summary>
        /// Dot product between an F32 vector <paramref name="a"/> and an F16
        /// vector <paramref name="b"/>. Each F16 element is converted to F32
        /// inside the inner loop using <see cref="BitConverter.UInt16BitsToHalf"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotF32F16(float* a, ushort* b, int n)
        {
            float sum = 0;
            for (int i = 0; i < n; i++)
                sum += a[i] * (float)BitConverter.UInt16BitsToHalf(b[i]);
            return sum;
        }

        /// <summary>Four-way dot product variant of <see cref="DotF32F16"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dot4F32F16(float* a0, float* a1, float* a2, float* a3,
            ushort* b, int n,
            out float r0, out float r1, out float r2, out float r3)
        {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < n; i++)
            {
                float bi = (float)BitConverter.UInt16BitsToHalf(b[i]);
                s0 += a0[i] * bi;
                s1 += a1[i] * bi;
                s2 += a2[i] * bi;
                s3 += a3[i] * bi;
            }
            r0 = s0; r1 = s1; r2 = s2; r3 = s3;
        }

        /// <summary>
        /// Read F16 source values, scale by <paramref name="weight"/> (F32),
        /// and accumulate into F32 destination buffer.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScaleAddF16(float* dst, ushort* src, float weight, int n)
        {
            for (int i = 0; i < n; i++)
                dst[i] += weight * (float)BitConverter.UInt16BitsToHalf(src[i]);
        }

        /// <summary>Four-way scale-add variant of <see cref="ScaleAddF16"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScaleAdd4F16(float* d0, float* d1, float* d2, float* d3,
            ushort* src, float w0, float w1, float w2, float w3, int n)
        {
            for (int i = 0; i < n; i++)
            {
                float s = (float)BitConverter.UInt16BitsToHalf(src[i]);
                d0[i] += w0 * s;
                d1[i] += w1 * s;
                d2[i] += w2 * s;
                d3[i] += w3 * s;
            }
        }

        public static void SelectTopKInPlace(ReadOnlySpan<float> values, int k, Span<int> indices)
        {
            if (k < 0)
                throw new ArgumentOutOfRangeException(nameof(k));
            if (indices.Length < k)
                throw new ArgumentException("The output indices span is shorter than k.", nameof(indices));
            if (k == 0)
                return;

            Span<float> topVals = stackalloc float[k];
            for (int i = 0; i < k; i++)
            {
                topVals[i] = float.NegativeInfinity;
                indices[i] = -1;
            }

            for (int i = 0; i < values.Length; i++)
            {
                int minIdx = 0;
                for (int j = 1; j < k; j++)
                {
                    if (topVals[j] < topVals[minIdx])
                        minIdx = j;
                }

                if (values[i] > topVals[minIdx])
                {
                    topVals[minIdx] = values[i];
                    indices[minIdx] = i;
                }
            }
        }

        public static void SelectTopKInPlace(float[] values, int n, int k, int[] indices)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            if ((uint)n > (uint)values.Length)
                throw new ArgumentOutOfRangeException(nameof(n));

            SelectTopKInPlace(values.AsSpan(0, n), k, indices);
        }

        public static void SelectTopKInPlace(float* values, int n, int k, int[] indices)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            if (n < 0)
                throw new ArgumentOutOfRangeException(nameof(n));

            SelectTopKInPlace(new ReadOnlySpan<float>(values, n), k, indices);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x)
        {
            if (x >= 0)
            {
                float e = MathF.Exp(-x);
                return 1.0f / (1.0f + e);
            }

            float en = MathF.Exp(x);
            return en / (1.0f + en);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SiLU(float x) => x * Sigmoid(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Softplus(float x)
        {
            if (x > 20f)
                return x;
            if (x < -20f)
                return MathF.Exp(x);
            return MathF.Log(1.0f + MathF.Exp(x));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplySiLUInPlace(Span<float> data, Span<float> scratch)
        {
            System.Numerics.Tensors.TensorPrimitives.Sigmoid(data, scratch);
            System.Numerics.Tensors.TensorPrimitives.Multiply(data, scratch, data);
        }

        public static void ReluSquaredInPlace(float* data, long count)
        {
            int vLen = Vector<float>.Count;
            var zero = Vector<float>.Zero;
            long i = 0;
            for (; i <= count - vLen; i += vLen)
            {
                var v = LoadVector(data + i);
                var mask = Vector.GreaterThan(v, zero);
                StoreVector(data + i, Vector.ConditionalSelect(mask, v * v, zero));
            }
            for (; i < count; i++)
            {
                float v = data[i];
                data[i] = v > 0f ? v * v : 0f;
            }
        }

        public static void ReluSquaredInPlace(Tensor tensor)
        {
            ReluSquaredInPlace(GetFloatPointer(tensor), tensor.ElementCount());
        }
    }
}
