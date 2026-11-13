// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Xml;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp
{
	public class TensorApplyCPU
	{
        private const int ParallelWorkThreshold = 1 << 15;

        #region Tensor iteration methods
        unsafe public delegate void Apply1KernelFunction(float* x);
		unsafe public delegate void Apply2KernelFunction(float* x, float* y);
		unsafe public delegate void Apply3KernelFunction(float* x, float* y, float* z);
		unsafe public delegate void Apply4KernelFunction(float* x, float* y, float* z, float* k);
		unsafe public delegate void Apply5KernelFunction(float* x, float* y, float* z, float* k, float* l);
		unsafe public delegate void ApplyDim2KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY);
		unsafe public delegate void ApplyDim3KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY, float* z, long sizeZ, long stridesZ);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static bool TryGetContiguousFloat(Tensor tensor, out float* ptr, out int length)
        {
            if (tensor.ElementType == DType.Float32 && tensor.IsContiguous() && tensor.ElementCount() <= int.MaxValue)
            {
                ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                length = (int)tensor.ElementCount();
                return true;
            }

            ptr = null;
            length = 0;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static bool TryGetContiguousRows(Tensor tensor, out float* ptr, out int rows, out int cols)
        {
            if (tensor.ElementType == DType.Float32 && tensor.IsContiguous() && tensor.ElementCount() <= int.MaxValue && tensor.Sizes[^1] <= int.MaxValue)
            {
                cols = (int)tensor.Sizes[^1];
                int elementCount = (int)tensor.ElementCount();
                if (cols > 0 && elementCount % cols == 0)
                {
                    ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                    rows = elementCount / cols;
                    return true;
                }
            }

            ptr = null;
            rows = 0;
            cols = 0;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldParallelize(int outerWork, int innerWork)
        {
            return outerWork > 1 && (long)outerWork * innerWork >= ParallelWorkThreshold;
        }

        unsafe private static void SiLUContiguous(float* resultPtr, float* srcPtr, int length)
        {
            ReadOnlySpan<float> input = new ReadOnlySpan<float>(srcPtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != srcPtr)
            {
                TensorPrimitives.Sigmoid(input, output);
                TensorPrimitives.Multiply(input, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> sigmoid = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(input, sigmoid);
                TensorPrimitives.Multiply(input, sigmoid, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        unsafe private static void SiLUMulContiguous(float* resultPtr, float* gatePtr, float* upPtr, int length)
        {
            ReadOnlySpan<float> gate = new ReadOnlySpan<float>(gatePtr, length);
            ReadOnlySpan<float> up = new ReadOnlySpan<float>(upPtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != gatePtr && resultPtr != upPtr)
            {
                TensorPrimitives.Sigmoid(gate, output);
                MultiplySiLUGateUp(gate, up, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> tmp = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(gate, tmp);
                MultiplySiLUGateUp(gate, up, tmp, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        private static void MultiplySiLUGateUp(
            ReadOnlySpan<float> gate,
            ReadOnlySpan<float> up,
            ReadOnlySpan<float> sigmoid,
            Span<float> output)
        {
            int vectorSize = Vector<float>.Count;
            int i = 0;
            for (; i <= output.Length - vectorSize; i += vectorSize)
            {
                Vector<float> value =
                    new Vector<float>(gate.Slice(i)) *
                    new Vector<float>(sigmoid.Slice(i)) *
                    new Vector<float>(up.Slice(i));
                value.CopyTo(output.Slice(i));
            }

            for (; i < output.Length; i++)
            {
                output[i] = gate[i] * sigmoid[i] * up[i];
            }
        }

        unsafe private static void SigmoidMulContiguous(float* resultPtr, float* xPtr, float* gatePtr, int length)
        {
            ReadOnlySpan<float> x = new ReadOnlySpan<float>(xPtr, length);
            ReadOnlySpan<float> gate = new ReadOnlySpan<float>(gatePtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != xPtr && resultPtr != gatePtr)
            {
                TensorPrimitives.Sigmoid(gate, output);
                TensorPrimitives.Multiply(x, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> tmp = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(gate, tmp);
                TensorPrimitives.Multiply(x, tmp, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static Vector<float> LoadVec(float* ptr)
        {
            return Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static void StoreVec(float* ptr, Vector<float> value)
        {
            Unsafe.WriteUnaligned(ref *(byte*)ptr, value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static float DotContiguous(float* lhs, float* rhs, int length)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> acc = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                acc += LoadVec(lhs + i) * LoadVec(rhs + i);
            }

            float sum = Vector.Sum(acc);
            for (; i < length; i++)
            {
                sum += lhs[i] * rhs[i];
            }

            return sum;
        }

		unsafe static void Apply1(Tensor tensor1, Apply1KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd(); tensor1Iter.BlockStep())
				{
					func(tensor1Iter.data);
				}

			} while (tensor1Iter.NextBlock());
		}


		unsafe static void Apply2(Tensor tensor1, Tensor tensor2, Apply2KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd(); tensor1Iter.BlockStep(), tensor2Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}


		unsafe static void Apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Apply3KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd();
						tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}


		unsafe static void Apply4(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Apply4KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);
			float* buffer4 = (float*)CpuNativeHelpers.GetBufferStart(tensor4);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory);
			TensorIterState tensor4Iter = new TensorIterState(buffer4, tensor4.DimensionCount, tensor4.SizesMemory, tensor4.StridesMemory);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd();
					tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data, tensor4Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock());
		}


		unsafe static void Apply5(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Tensor tensor5, Apply5KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);
			float* buffer4 = (float*)CpuNativeHelpers.GetBufferStart(tensor4);
			float* buffer5 = (float*)CpuNativeHelpers.GetBufferStart(tensor5);


			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, step);
			TensorIterState tensor4Iter = new TensorIterState(buffer4, tensor4.DimensionCount, tensor4.SizesMemory, tensor4.StridesMemory, step);
			TensorIterState tensor5Iter = new TensorIterState(buffer5, tensor5.DimensionCount, tensor5.SizesMemory, tensor5.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd() && !tensor5Iter.ReachedBlockEnd();
					tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep(), tensor5Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data, tensor4Iter.data, tensor5Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock() && tensor5Iter.NextBlock());
		}


		unsafe static void ApplyDim2(Tensor tensor1, Tensor tensor2, int iterationDim, ApplyDim2KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}




		unsafe static void ApplyDim3(Tensor tensor1, Tensor tensor2, Tensor tensor3, int iterationDim, ApplyDim3KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, iterationDim);
			TensorDimIterState tensor3Iter = new TensorDimIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride,
					tensor3Iter.data, tensor3Iter.size, tensor3Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}


        #endregion

        // True iff all three tensors are 2D contiguous float32 with matching
        // outer shape (i.e. a typical [N,D] gather/scatter layout). The fast
        // paths below bypass the generic TensorDimIterState walk: they iterate
        // rows in the outer loop (cache-friendly writes) and detect the
        // embedding-style "all indices in a row equal" pattern to emit a
        // single row memcpy instead of D scalar copies.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryGetGatherScatter2DFast(
            Tensor result, Tensor src, Tensor indices, int dim,
            out float* rPtr, out float* sPtr, out float* iPtr,
            out long N, out long D, out long otherDim)
        {
            rPtr = sPtr = iPtr = null;
            N = D = otherDim = 0;

            if (result == null || src == null || indices == null) return false;
            if (result.DimensionCount != 2 || src.DimensionCount != 2 || indices.DimensionCount != 2) return false;
            if (dim != 0 && dim != 1) return false;
            if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32) return false;
            if (indices.ElementType != DType.Float32) return false;
            if (!result.IsContiguous() || !src.IsContiguous() || !indices.IsContiguous()) return false;

            // Shape preconditions match the public API: result.shape == indices.shape,
            // and src/result agree on every dim except `dim`.
            if (result.Sizes[0] != indices.Sizes[0] || result.Sizes[1] != indices.Sizes[1]) return false;
            int otherIdx = dim == 0 ? 1 : 0;
            if (src.Sizes[otherIdx] != result.Sizes[otherIdx]) return false;

            N = result.Sizes[0];
            D = result.Sizes[1];
            otherDim = src.Sizes[dim];

            rPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
            sPtr = (float*)CpuNativeHelpers.GetBufferStart(src);
            iPtr = (float*)CpuNativeHelpers.GetBufferStart(indices);
            return true;
        }

        unsafe public static void Gather(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long sSize))
            {
                if (dim == 0)
                {
                    // result[i, j] = src[indices[i, j], j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* resRow = rPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= sSize)
                            throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{firstIdx}', sSize = '{sSize}'");

                        // Detect embedding-style uniform-row indices and emit
                        // a single row memcpy.
                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
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
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= sSize)
                                    throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize}'");
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
                        float* srcRow = sPtr + i * srcCols;
                        float* indRow = iPtr + i * D;
                        float* resRow = rPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= srcCols)
                                throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{srcCols}'");
                            resRow[j] = srcRow[idx];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize, long rStride,
				float* sData, long sSize2, long sStride,
				float* iData, long iSize, long iStride)
			{
				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= sSize2) { throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize2}'"); }

					*(rData + i * rStride) = sData[idx * sStride];
				}
			}

			ApplyDim3(result, src, indices, dim, func);
		}



		unsafe public static void Scatter(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] = src[i, j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{firstIdx}', rSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
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
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'");
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
                        float* resRow = rPtr + i * resCols;
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] = srcRow[j];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize2, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{

				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize2) { throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize2}'"); }

					rData[idx * rStride] = *(sData + i * sStride);
				}

			}

			ApplyDim3(result, src, indices, dim, func);
		}

		unsafe public static void ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] += src[i, j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{firstIdx}', rSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            // Vectorized add for the destination row.
                            int vectorSize = Vector<float>.Count;
                            float* dst = rPtr + firstIdx * D;
                            long j = 0;
                            for (; j <= D - vectorSize; j += vectorSize)
                            {
                                Vector<float> a = LoadVec(dst + j);
                                Vector<float> b = LoadVec(srcRow + j);
                                StoreVec(dst + j, a + b);
                            }
                            for (; j < D; j++)
                                dst[j] += srcRow[j];
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'");
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
                        float* resRow = rPtr + i * resCols;
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] += srcRow[j];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize2, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{

				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize2) { throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize2}'"); }

					rData[idx * rStride] += *(sData + i * sStride);
				}

			}

			ApplyDim3(result, src, indices, dim, func);
		}

		unsafe public static void ScatterFill(Tensor result, float value, int dim, Tensor indices)
		{
            // 2D-contig fast path for ScatterFill (no src tensor).
            if (result != null && indices != null
                && result.DimensionCount == 2 && indices.DimensionCount == 2
                && (dim == 0 || dim == 1)
                && result.ElementType == DType.Float32 && indices.ElementType == DType.Float32
                && result.IsContiguous() && indices.IsContiguous()
                && indices.Sizes[1 - dim] == result.Sizes[1 - dim])
            {
                long N = indices.Sizes[0];
                long D = indices.Sizes[1];
                long otherDim = result.Sizes[dim];

                float* rPtrFast = (float*)CpuNativeHelpers.GetBufferStart(result);
                float* iPtrFast = (float*)CpuNativeHelpers.GetBufferStart(indices);

                if (dim == 0)
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtrFast + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= otherDim)
                                throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{otherDim}'");
                            rPtrFast[idx * resCols + j] = value;
                        }
                    }
                }
                else // dim == 1
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        float* resRow = rPtrFast + i * resCols;
                        float* indRow = iPtrFast + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] = value;
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize, long rStride, float* iData, long iSize, long iStride)
			{
				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize) { throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{rSize}'"); }

					rData[idx * rStride] = value;
				}

			}

			// the assumption here is that the result and indices are not null. The compiler can't guarantee this because of the null tests in the if() statement above.
			ApplyDim2(result!, indices!, dim, func);
		}



		unsafe public static void Fill(Tensor result, float value)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length))
            {
                new Span<float>(resultPtr, length).Fill(value);
                return;
            }

            // The generic Apply1 path treats every element as 4 bytes, so
            // routing a Float16 (or block-quantized Q8_0) tensor through it
            // walks past the storage buffer and surfaces as an
            // AccessViolationException during KV-cache zero-init. Mirror the
            // GGML backend's contiguous F16/Q8_0 fast paths so non-F32 caches
            // can be filled safely on the managed CPU backend.
            if (result.ElementType == DType.Float16 && IsContiguousNonNarrowed(result))
            {
                ushort halfBits = BitConverter.HalfToUInt16Bits((System.Half)value);
                ushort* halfBuffer = (ushort*)CpuNativeHelpers.GetBufferStart(result);
                long elementCount = result.ElementCount();
                if (halfBits == 0)
                {
                    long offset = 0;
                    while (offset < elementCount)
                    {
                        int slice = (int)Math.Min(elementCount - offset, int.MaxValue);
                        new Span<ushort>(halfBuffer + offset, slice).Clear();
                        offset += slice;
                    }
                }
                else
                {
                    for (long i = 0; i < elementCount; i++)
                        halfBuffer[i] = halfBits;
                }
                return;
            }

            if (result.ElementType == DType.Q8_0 && IsContiguousNonNarrowed(result))
            {
                if (value != 0f)
                    throw new NotSupportedException("Fill on Q8_0 tensors only supports value=0 (cache reset).");
                long byteLength = DTypeExtensions.Q8_0Bytes(result.ElementCount());
                byte* byteBuffer = (byte*)CpuNativeHelpers.GetBufferStart(result);
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
                throw new NotSupportedException(
                    $"Fill on {result.ElementType} tensors requires a contiguous, non-narrowed layout.");

			unsafe void func(float* r)
			{
				*r = value;
			}

			Apply1(result, func);
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


		unsafe public static void Clamp(Tensor result, Tensor src, float min, float max)
		{
			unsafe void func(float* r, float* s)
			{
				*r = clamp(*s, min, max);
			}
			Apply2(result, src, func);
		}


		unsafe public static void Copy(Tensor result, Tensor src)
		{
            if (result.IsContiguous() && src.IsContiguous() &&
                result.ElementType == src.ElementType &&
                result.ElementCount() == src.ElementCount())
            {
                long byteCount = result.ElementCount() * result.ElementType.Size();
                if (byteCount <= int.MaxValue)
                {
                    byte* srcBytes = (byte*)CpuNativeHelpers.GetBufferStart(src);
                    byte* resultBytes = (byte*)CpuNativeHelpers.GetBufferStart(result);
                    new ReadOnlySpan<byte>(srcBytes, (int)byteCount).CopyTo(new Span<byte>(resultBytes, (int)byteCount));
                }
                else
                {
                    Buffer.MemoryCopy(
                        CpuNativeHelpers.GetBufferStart(src).ToPointer(),
                        CpuNativeHelpers.GetBufferStart(result).ToPointer(),
                        byteCount,
                        byteCount);
                }
                return;
            }

			// The strided paths below walk both buffers as 4-byte elements: right for F32 and Int32,
			// a buffer overrun for F16 (and a reinterpretation across types). Those go element by element.
			if (result.ElementType != src.ElementType || result.ElementType.Size() != 4)
			{
				CopyConverting(result, src);
				return;
			}

			if (TryCopyStridedSimd(result, src))
			{
				return;
			}

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);
					vecS.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = *s;
				}
				Apply2(result, src, func);
			}
		}


		/// <summary>
		/// A copy between same-shaped views of any strides that converts F32 and F16 (unit-stride rows
		/// through the vectorized converters); other element types are refused rather than reinterpreted.
		/// </summary>
		unsafe static void CopyConverting(Tensor result, Tensor src)
		{
			DType dt = result.ElementType, st = src.ElementType;
			if ((dt != DType.Float32 && dt != DType.Float16) || (st != DType.Float32 && st != DType.Float16))
				throw new NotSupportedException($"CPU copy from {st} to {dt} between strided views");
			int rank = result.DimensionCount;
			if (rank != src.DimensionCount)
				throw new NotSupportedException("CPU converting copy between views of different ranks");
			for (int d = 0; d < rank; d++)
				if (result.Sizes[d] != src.Sizes[d])
					throw new NotSupportedException("CPU converting copy between views of different sizes");
			if (rank == 0)
				return;
			long inner = result.Sizes[rank - 1];
			long dInner = result.Strides[rank - 1], sInner = src.Strides[rank - 1];
			byte* d0 = (byte*)CpuNativeHelpers.GetBufferStart(result), s0 = (byte*)CpuNativeHelpers.GetBufferStart(src);
			int de = dt.Size(), se = st.Size();
			long[] index = new long[Math.Max(1, rank - 1)];
			long rows = inner == 0 ? 0 : result.ElementCount() / inner;
			for (long r = 0; r < rows; r++)
			{
				long dOff = 0, sOff = 0;
				for (int i = 0; i < rank - 1; i++)
				{
					dOff += index[i] * result.Strides[i];
					sOff += index[i] * src.Strides[i];
				}
				byte* dRow = d0 + dOff * de, sRow = s0 + sOff * se;
				if (dInner == 1 && sInner == 1 && dt == st)
					Buffer.MemoryCopy(sRow, dRow, inner * de, inner * de);
				else if (dInner == 1 && sInner == 1 && dt == DType.Float16)
					TensorComputePrimitives.F32ToF16((ushort*)dRow, (float*)sRow, checked((int)inner));
				else if (dInner == 1 && sInner == 1)
					TensorComputePrimitives.F16ToF32((float*)dRow, (ushort*)sRow, checked((int)inner));
				else
				{
					for (long j = 0; j < inner; j++)
					{
						byte* sp = sRow + j * sInner * se, dp = dRow + j * dInner * de;
						float v = st == DType.Float32 ? *(float*)sp : (float)BitConverter.UInt16BitsToHalf(*(ushort*)sp);
						if (dt == DType.Float32) *(float*)dp = v;
						else *(ushort*)dp = BitConverter.HalfToUInt16Bits((System.Half)v);
					}
				}
				for (int i = rank - 2; i >= 0; i--)
				{
					if (++index[i] < result.Sizes[i])
						break;
					index[i] = 0;
				}
			}
		}

		/// <summary>
		/// Strided F32 copy between same-shaped views (head split/merge, narrowed rows, permutes).
		/// Dimensions both operands can walk as one are collapsed first, so a [seq, heads, hd] ->
		/// [heads, seq, hd] permute becomes a loop of hd-float row copies; the outer index space
		/// is split on CpuParallel by element count. The generic iterator this replaces paid a
		/// delegate call per 8 elements (or per element when the last dim is not a multiple of 8).
		/// </summary>
		unsafe static bool TryCopyStridedSimd(Tensor result, Tensor src)
		{
			if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32 ||
				result.DimensionCount != src.DimensionCount)
			{
				return false;
			}

			int dims = result.DimensionCount;
			for (int d = 0; d < dims; d++)
			{
				if (result.Sizes[d] != src.Sizes[d])
					return false;
			}

			long[] size = new long[Math.Max(1, dims)];
			long[] rs = new long[Math.Max(1, dims)];
			long[] ss = new long[Math.Max(1, dims)];
			int nd = 0;
			for (int d = 0; d < dims; d++)
			{
				long sz = result.Sizes[d];
				if (sz == 1)
					continue;
				if (sz == 0)
					return true;

				long r = result.Strides[d], s = src.Strides[d];
				if (nd > 0 && rs[nd - 1] == r * sz && ss[nd - 1] == s * sz)
				{
					size[nd - 1] *= sz;
					rs[nd - 1] = r;
					ss[nd - 1] = s;
				}
				else
				{
					size[nd] = sz;
					rs[nd] = r;
					ss[nd] = s;
					nd++;
				}
			}

			float* rBase = (float*)CpuNativeHelpers.GetBufferStart(result);
			float* sBase = (float*)CpuNativeHelpers.GetBufferStart(src);
			if (nd == 0)
			{
				*rBase = *sBase;
				return true;
			}

			long inner = size[nd - 1];
			long rInner = rs[nd - 1], sInner = ss[nd - 1];
			int outerDims = nd - 1;
			long outer = 1;
			for (int d = 0; d < outerDims; d++)
				outer *= size[d];

			CpuParallel.ForRange(outer, Math.Max(1, (64 * 1024) / inner), [MethodImpl(MethodImplOptions.AggressiveOptimization)] (start, end) =>
			{
				long[] idx = new long[Math.Max(1, outerDims)];
				long ro = 0, so = 0, rem = start;
				for (int d = outerDims - 1; d >= 0; d--)
				{
					idx[d] = rem % size[d];
					rem /= size[d];
					ro += idx[d] * rs[d];
					so += idx[d] * ss[d];
				}

				for (long o = start; o < end; o++)
				{
					float* rp = rBase + ro;
					float* sp = sBase + so;
					if (rInner == 1 && sInner == 1)
					{
						Buffer.MemoryCopy(sp, rp, inner * sizeof(float), inner * sizeof(float));
					}
					else
					{
						for (long i = 0; i < inner; i++)
							rp[i * rInner] = sp[i * sInner];
					}

					for (int d = outerDims - 1; d >= 0; d--)
					{
						ro += rs[d];
						so += ss[d];
						if (++idx[d] < size[d])
							break;
						ro -= size[d] * rs[d];
						so -= size[d] * ss[d];
						idx[d] = 0;
					}
				}
			});

			return true;
		}

		unsafe public static void Sum(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float sum = 0.0f;
				for (long i = 0; i < sSize; ++i)
				{
					sum += s[i * sStride];
				}
				*r = sum;
			}
			ApplyDim2(result, src, dimension, func);
		}


		unsafe public static void Mean(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float sum = 0.0f;
				for (long i = 0; i < sSize; ++i)
				{
					sum += s[i * sStride];
				}
				*r = sum / sSize;
			}
			ApplyDim2(result, src, dimension, func);
		}


		unsafe public static void Argmax(Tensor resultIndices, Tensor src, int dimension)
		{

			unsafe void func(float* rIndVal, long rIndSize, long rIndStride,
				float* s, long sSize, long sStride)
			{
				float value = s[0];
				float index = 0;
				for (long i = 1; i < sSize; ++i)
				{
					float currentVal = s[i * sStride];
					if (currentVal > value)
					{
						value = currentVal;
						index = (float)i;
					}
				}
				*rIndVal = index;
			}

			ApplyDim2(resultIndices, src, dimension, func);
		}

		unsafe public static void Max(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float value = s[0];
				for (long i = 1; i < sSize; ++i)
				{
					value = Math.Max(value, s[i * sStride]);
				}
				*r = value;
			}

			ApplyDim2(result, src, dimension, func);
		}


		// result[r, c] = lhs[r, c] + rhs[c] for contiguous result/lhs and a
		// trailing-dimension rhs. Returns false when the operands are not that
		// shape, leaving the caller on its generic path.
		internal enum RowBroadcastOp { Add, Sub, Mul, Div }

		/// <summary>
		/// Elementwise <paramref name="lhs"/> OP <paramref name="rhs"/> where rhs is
		/// one row broadcast down every row of lhs (x[rows, cols] OP bias[cols]) — the
		/// shape a per-channel bias or scale takes. Apply3 below cannot express it: it
		/// advances all three iterators together and stops as soon as the shortest one
		/// runs out of blocks, so a [cols] rhs would touch row 0 only and leave the
		/// rest of the tensor silently unchanged.
		/// </summary>
		unsafe static bool TryRowBroadcast(Tensor result, Tensor lhs, Tensor rhs, RowBroadcastOp op)
		{
			if (result == null || lhs == null || rhs == null || lhs.DimensionCount < 2)
				return false;

			long cols = lhs.Sizes[lhs.DimensionCount - 1];
			if (cols <= 0 || rhs.ElementCount() != cols || lhs.ElementCount() == cols)
				return false;

			for (int i = 0; i < rhs.DimensionCount - 1; i++)
			{
				if (rhs.Sizes[i] != 1)
					return false;
			}

			if (result.DimensionCount != lhs.DimensionCount)
				return false;
			for (int i = 0; i < lhs.DimensionCount; i++)
			{
				if (result.Sizes[i] != lhs.Sizes[i])
					return false;
			}

			if (!TryGetContiguousFloat(result, out float* resultPtr, out int resultLength) ||
				!TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) ||
				!TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) ||
				resultLength != lhsLength || rhsLength != cols)
			{
				return false;
			}

			int colCount = (int)cols;
			int rows = lhsLength / colCount;
			CpuKernels.BinaryRowBroadcast(ToKernelOp(op), resultPtr, lhsPtr, rhsPtr, rows, colCount);
			return true;
		}

		private static CpuKernels.BinaryOp ToKernelOp(RowBroadcastOp op) => op switch
		{
			RowBroadcastOp.Add => CpuKernels.BinaryOp.Add,
			RowBroadcastOp.Sub => CpuKernels.BinaryOp.Sub,
			RowBroadcastOp.Mul => CpuKernels.BinaryOp.Mul,
			_ => CpuKernels.BinaryOp.Div,
		};

		/// <summary>
		/// SIMD + parallel tensor-tensor path (CpuKernels): equal-length contiguous operands, or the
		/// x[rows, cols] OP row[cols] broadcast. False leaves the caller on its previous loops.
		/// </summary>
		unsafe static bool TryBinarySimd(Tensor result, Tensor lhs, Tensor rhs, RowBroadcastOp op)
		{
			if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
				TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
				TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
				length == lhsLength && length == rhsLength)
			{
				CpuKernels.Binary(ToKernelOp(op), resultPtr, lhsPtr, rhsPtr, length);
				return true;
			}

			return TryRowBroadcast(result, lhs, rhs, op);
		}

		/// <summary>SIMD + parallel tensor-scalar path; scalarOnLeft computes value OP src.</summary>
		unsafe static bool TryScalarSimd(Tensor result, Tensor src, float value, CpuKernels.BinaryOp op, bool scalarOnLeft = false)
		{
			if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
				TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
				length == srcLength)
			{
				CpuKernels.BinaryScalar(op, resultPtr, srcPtr, value, length, scalarOnLeft);
				return true;
			}

			return false;
		}

		unsafe public static void Add(Tensor result, Tensor lhs, Tensor rhs)
		{
			if (TryBinarySimd(result, lhs, rhs, RowBroadcastOp.Add))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) + LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] + rhsPtr[i];
                }

                return;
            }

            // Row-broadcast (x[rows, cols] + bias[cols]) - see TryRowBroadcast.
            if (TryRowBroadcast(result, lhs, rhs, RowBroadcastOp.Add))
                return;

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft + vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = add(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}


		unsafe public static void Sub(Tensor result, Tensor lhs, Tensor rhs)
		{
			if (TryBinarySimd(result, lhs, rhs, RowBroadcastOp.Sub))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) - LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] - rhsPtr[i];
                }

                return;
            }

            // Row-broadcast (x[rows, cols] OP bias[cols]) - see TryRowBroadcast.
            if (TryRowBroadcast(result, lhs, rhs, RowBroadcastOp.Sub))
                return;


			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft - vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = *left - *right;
				}

				Apply3(result, lhs, rhs, func);
			}
		}

		unsafe public static void Add(Tensor result, Tensor src, float value)
		{
			if (TryScalarSimd(result, src, value, CpuKernels.BinaryOp.Add))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(srcPtr + i) + vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = srcPtr[i] + value;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS + vecV;
					vecR.CopyTo(spanR);

				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = add(*s, value);
				}

				Apply2(result, src, func);
			}
		}


		unsafe public static void Pow(Tensor result, Tensor src, float value)
		{
				unsafe void func(float* r, float* s)
				{
					*r = (float)Math.Pow(*s, value);
				}

				Apply2(result, src, func);			
		}


		unsafe public static void RSub(Tensor result, float value, Tensor src)
		{
			if (TryScalarSimd(result, src, value, CpuKernels.BinaryOp.Sub, scalarOnLeft: true))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, vecValue - LoadVec(srcPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = value - srcPtr[i];
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecV - vecS;
					vecR.CopyTo(spanR);

				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = value - *s;
				}

				Apply2(result, src, func);
			}
		}



		/// <summary>result = value / src (elementwise).</summary>
		unsafe internal static void RDiv(Tensor result, float value, Tensor src)
		{
			if (TryScalarSimd(result, src, value, CpuKernels.BinaryOp.Div, scalarOnLeft: true))
				return;

			unsafe void func(float* r, float* s)
			{
				*r = value / *s;
			}

			Apply2(result, src, func);
		}

		unsafe public static void Mul(Tensor result, Tensor src, float value)
		{
			if (TryScalarSimd(result, src, value, CpuKernels.BinaryOp.Mul))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(srcPtr + i) * vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = srcPtr[i] * value;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS * vecV;
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = mul(*s, value);
				}

				Apply2(result, src, func);
			}
		}


		unsafe public static void Div(Tensor result, Tensor lhs, float rhs)
		{
			if (TryScalarSimd(result, lhs, rhs, CpuKernels.BinaryOp.Div))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                length == lhsLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(rhs);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) / vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] / rhs;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(rhs);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS / vecV;
					vecR.CopyTo(spanR);
				}

				Apply2(result, lhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = div(*s, rhs);
				}

				Apply2(result, lhs, func);
			}
		}

		unsafe public static void Mul(Tensor result, Tensor lhs, Tensor rhs)
		{
			if (TryBinarySimd(result, lhs, rhs, RowBroadcastOp.Mul))
				return;

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) * LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] * rhsPtr[i];
                }

                return;
            }

            // Row-broadcast (x[rows, cols] OP bias[cols]) - see TryRowBroadcast.
            if (TryRowBroadcast(result, lhs, rhs, RowBroadcastOp.Mul))
                return;


			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft * vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = mul(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}


		unsafe public static void Div(Tensor result, Tensor lhs, Tensor rhs)
		{
			if (TryBinarySimd(result, lhs, rhs, RowBroadcastOp.Div))
				return;

			// Row-broadcast (x[rows, cols] / bias[cols]) - see TryRowBroadcast.
			if (TryRowBroadcast(result, lhs, rhs, RowBroadcastOp.Div))
				return;

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft / vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = div(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}

		unsafe static public void Relu(Tensor result, Tensor src)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = Vector.Max(vecS, Vector<float>.Zero);
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = relu(*s);
				};

				Apply2(result, src, func);
			}
		}


		unsafe static public void Abs(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = Math.Abs(*s);
			}
			Apply2(result, src, func);
		}

		unsafe static public void Neg(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = -(*s);
			}
			Apply2(result, src, func);
		}

		unsafe static public void Sqrt(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = (float)Math.Sqrt(*s);
			}
			Apply2(result, src, func);
		}

		unsafe static public void Rsqrt(Tensor result, Tensor src)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = Vector<float>.One / Vector.SquareRoot(vecS);
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = (float)(1.0 / Math.Sqrt(*s));
				};

				Apply2(result, src, func);
			}
		}



		/// <summary>Contiguous equal-length operands for a CpuKernels unary kernel.</summary>
		unsafe static bool TryUnarySimd(Tensor result, Tensor src, out float* resultPtr, out float* srcPtr, out int length)
		{
			srcPtr = null;
			if (TryGetContiguousFloat(result, out resultPtr, out length) &&
				TryGetContiguousFloat(src, out srcPtr, out int srcLength) &&
				length == srcLength)
			{
				return true;
			}

			resultPtr = null;
			length = 0;
			return false;
		}

		/// <summary>Contiguous equal-length operands for a CpuKernels gated (two-input) kernel.</summary>
		unsafe static bool TryGatedSimd(Tensor result, Tensor a, Tensor b, out float* resultPtr, out float* aPtr, out float* bPtr, out int length)
		{
			aPtr = null;
			bPtr = null;
			if (TryGetContiguousFloat(result, out resultPtr, out length) &&
				TryGetContiguousFloat(a, out aPtr, out int aLength) &&
				TryGetContiguousFloat(b, out bPtr, out int bLength) &&
				length == aLength && length == bLength)
			{
				return true;
			}

			resultPtr = null;
			length = 0;
			return false;
		}

		unsafe static public void Sigmoid(Tensor result, Tensor src)
		{
			if (TryUnarySimd(result, src, out float* rSimd, out float* sSimd, out int nSimd))
			{
				CpuKernels.Sigmoid(rSimd, sSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                TensorPrimitives.Sigmoid(new ReadOnlySpan<float>(srcPtr, length), new Span<float>(resultPtr, length));
                return;
            }


			unsafe void func(float* r, float* s)
			{
				*r = sigmoid(*s);
			};

			Apply2(result, src, func);
		}






		unsafe static public void Tanh(Tensor result, Tensor src)
		{
			if (TryUnarySimd(result, src, out float* rSimd, out float* sSimd, out int nSimd))
			{
				CpuKernels.Tanh(rSimd, sSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = MathF.Tanh(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = MathF.Tanh(*s);
			};

			Apply2(result, src, func);
		}


		unsafe static public void Log(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = MathF.Log(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = MathF.Log(*s);
			};

			Apply2(result, src, func);
		}

        unsafe static public void Exp(Tensor result, Tensor src)
        {
            if (TryUnarySimd(result, src, out float* rSimd, out float* sSimd, out int nSimd))
            {
                CpuKernels.Exp(rSimd, sSimd, nSimd);
                return;
            }

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                TensorPrimitives.Exp(new ReadOnlySpan<float>(srcPtr, length), new Span<float>(resultPtr, length));
                return;
            }

            unsafe void func(float* r, float* s)
            {
                *r = MathF.Exp(*s);
            };

            Apply2(result, src, func);
        }




		unsafe static public void SiLU(Tensor result, Tensor src)
		{
			if (TryUnarySimd(result, src, out float* rSimd, out float* sSimd, out int nSimd))
			{
				CpuKernels.SiLU(rSimd, sSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                SiLUContiguous(resultPtr, srcPtr, length);
                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = SiLU(*s);
			};

			Apply2(result, src, func);
		}

		unsafe static public void GELU(Tensor result, Tensor src)
		{
			if (TryUnarySimd(result, src, out float* rSimd, out float* sSimd, out int nSimd))
			{
				CpuKernels.Gelu(rSimd, sSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = GELU(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = GELU(*s);
			};
			Apply2(result, src, func);
		}

		unsafe static public void GELUMul(Tensor result, Tensor gate, Tensor up)
		{
			if (TryGatedSimd(result, gate, up, out float* rSimd, out float* gSimd, out float* uSimd, out int nSimd))
			{
				CpuKernels.GeluMul(rSimd, gSimd, uSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                TryGetContiguousFloat(up, out float* upPtr, out int upLength) &&
                length == gateLength && length == upLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = GELU(gatePtr[i]) * upPtr[i];
                }

                return;
            }

			unsafe void func(float* r, float* g, float* u)
			{
				*r = GELU(*g) * *u;
			}
			Apply3(result, gate, up, func);
		}

		unsafe static public void SiLUMul(Tensor result, Tensor gate, Tensor up)
		{
			if (TryGatedSimd(result, gate, up, out float* rSimd, out float* gSimd, out float* uSimd, out int nSimd))
			{
				CpuKernels.SiLUMul(rSimd, gSimd, uSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                TryGetContiguousFloat(up, out float* upPtr, out int upLength) &&
                length == gateLength && length == upLength)
            {
                SiLUMulContiguous(resultPtr, gatePtr, upPtr, length);
                return;
            }

			unsafe void func(float* r, float* g, float* u)
			{
				*r = SiLU(*g) * *u;
			}
			Apply3(result, gate, up, func);
		}

		/// <summary>
		/// result = clamp(up, -limit, +limit) * SiLU(min(gate, limit)). limit &lt;= 0
		/// disables the clamp (plain SiLUMul). Aliasing result == gate is supported,
		/// which is how the MoE FFN writes the activated hidden state back in place.
		/// </summary>
		unsafe static public void SiLUMulClamp(Tensor result, Tensor gate, Tensor up, float limit)
		{
			if (!(limit > 0.0f))
			{
				SiLUMul(result, gate, up);
				return;
			}

			if (TryGatedSimd(result, gate, up, out float* rSimd, out float* gSimd, out float* uSimd, out int nSimd))
			{
				CpuKernels.SiLUMulClamp(rSimd, gSimd, uSimd, nSimd, limit);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                TryGetContiguousFloat(up, out float* upPtr, out int upLength) &&
                length == gateLength && length == upLength)
            {
                SiLUMulClampContiguous(resultPtr, gatePtr, upPtr, length, limit);
                return;
            }

			float lim = limit;
			unsafe void func(float* r, float* g, float* u)
			{
				float gv = MathF.Min(*g, lim);
				float uv = MathF.Min(MathF.Max(*u, -lim), lim);
				*r = SiLU(gv) * uv;
			}
			Apply3(result, gate, up, func);
		}

		unsafe private static void SiLUMulClampContiguous(float* resultPtr, float* gatePtr, float* upPtr, int length, float limit)
		{
			// Clamp both operands into scratch first so the vectorized
			// sigmoid/multiply below is the same code path SiLUMul uses.
			float[] gateRent = ArrayPool<float>.Shared.Rent(length);
			float[] upRent = ArrayPool<float>.Shared.Rent(length);
			try
			{
				Span<float> g = gateRent.AsSpan(0, length);
				Span<float> u = upRent.AsSpan(0, length);
				TensorPrimitives.Min(new ReadOnlySpan<float>(gatePtr, length), limit, g);
				TensorPrimitives.Max(new ReadOnlySpan<float>(upPtr, length), -limit, u);
				TensorPrimitives.Min(u, limit, u);

				Span<float> output = new Span<float>(resultPtr, length);
				TensorPrimitives.Sigmoid(g, output);
				MultiplySiLUGateUp(g, u, output, output);
			}
			finally
			{
				ArrayPool<float>.Shared.Return(gateRent);
				ArrayPool<float>.Shared.Return(upRent);
			}
		}

		unsafe static public void SigmoidMul(Tensor result, Tensor x, Tensor gate)
		{
			if (TryGatedSimd(result, x, gate, out float* rSimd, out float* xSimd, out float* gSimd, out int nSimd))
			{
				CpuKernels.SigmoidMul(rSimd, xSimd, gSimd, nSimd);
				return;
			}

            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(x, out float* xPtr, out int xLength) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                length == xLength && length == gateLength)
            {
                SigmoidMulContiguous(resultPtr, xPtr, gatePtr, length);
                return;
            }

			unsafe void func(float* r, float* a, float* b)
			{
				float sig = 1.0f / (1.0f + MathF.Exp(-*b));
				*r = *a * sig;
			}
			Apply3(result, x, gate, func);
		}


		unsafe static public void SiLUD(Tensor result, Tensor srcW, Tensor resG)
		{
			unsafe void func(float* r, float* y, float* x)
			{
				*r = SiLUD(*y, *x);
			}

			Apply3(result, srcW, resG, func);
		}

		unsafe static public void AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
		{
			unsafe void func(float* r, float* x, float* w, float* g)
			{
				*r = AddSiLUD(*x, *w, *g);
			}

			Apply4(result, srcG, srcW, resG, func);
		}



        unsafe static public void LeakyReLU(Tensor result, Tensor src)
        {
            unsafe void func(float* r, float* s)
            {
                *r = LeakyReLU(*s);
            };

            Apply2(result, src, func);
        }


        unsafe static public void LeakyReLUD(Tensor result, Tensor srcW, Tensor resG)
        {
            unsafe void func(float* r, float* y, float* x)
            {
                *r = LeakyReLUD(*y, *x);
            }

            Apply3(result, srcW, resG, func);
        }

        unsafe static public void AddLeakyReLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
        {
            unsafe void func(float* r, float* x, float* w, float* g)
            {
                *r = AddLeakyReLUD(*x, *w, *g);
            }

            Apply4(result, srcG, srcW, resG, func);
        }


        unsafe static public void AddMulV(Tensor result, Tensor srcX, Tensor srcY, float val)
		{
			unsafe void func(float* r, float* x, float* y)
			{
				*r = *x + (*y * val);
			}

			Apply3(result, srcX, srcY, func);
		}


		unsafe static public void MulMulAdd(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && srcX.Strides[^1] == 1 && srcY.Strides[^1] == 1 && srcZ.Strides[^1] == 1 && srcW.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* x, float* y, float* z, float* w)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanX = new Span<float>(x, vectorSize);
					Span<float> spanY = new Span<float>(y, vectorSize);
					Span<float> spanZ = new Span<float>(z, vectorSize);
					Span<float> spanW = new Span<float>(w, vectorSize);

					Vector<float> vecX = new Vector<float>(spanX);
					Vector<float> vecY = new Vector<float>(spanY);
					Vector<float> vecZ = new Vector<float>(spanZ);
					Vector<float> vecW = new Vector<float>(spanW);

					Vector<float> vecR = vecX * vecY + vecZ * vecW;
					vecR.CopyTo(spanR);
				}

				Apply5(result, srcX, srcY, srcZ, srcW, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* x, float* y, float* z, float* w)
				{
					*r = mulmuladd(*x, *y, *z, *w);
				}

				Apply5(result, srcX, srcY, srcZ, srcW, func);
			}
		}




		unsafe static public void AddMul(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = addmul(*x, *y, *z);
			}

			Apply4(result, srcX, srcY, srcZ, func);
		}


		unsafe static public void AddDiv(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = adddiv(*x, *y, *z);
			}

			Apply4(result, srcX, srcY, srcZ, func);
		}




		unsafe static public void RepeatInterleave(float* dst, float* src, int sliceCount, int repeats, int sliceSize)
		{
            long sliceBytes = (long)sliceSize * sizeof(float);

            void CopySlice(int i)
            {
                float* srcSlice = src + i * sliceSize;
                float* dstSlice = dst + i * repeats * sliceSize;
                for (int r = 0; r < repeats; r++)
                {
                    Buffer.MemoryCopy(srcSlice, dstSlice + r * sliceSize, sliceBytes, sliceBytes);
                }
            }

            CpuParallel.ForRange(sliceCount, Math.Max(1, (64 * 1024) / Math.Max(1, (long)repeats * sliceSize)), (s, e) =>
            {
                for (long i = s; i < e; i++) CopySlice((int)i);
            });
		}

	unsafe static public void AddCausalMask(float* data, int totalRows, int cols, int seqLen, int startPos, float maskedValue)
	{
        void MaskRow(int row)
        {
            int t = row % seqLen;
            int threshold = startPos + t;
            int sStart = Math.Max(0, threshold + 1);
            if (sStart >= cols)
            {
                return;
            }

            float* rowPtr = data + row * cols;
            if (float.IsNegativeInfinity(maskedValue))
            {
                new Span<float>(rowPtr + sStart, cols - sStart).Fill(float.NegativeInfinity);
                return;
            }

            for (int s = sStart; s < cols; s++)
            {
                rowPtr[s] += maskedValue;
            }
        }

        CpuParallel.ForRange(totalRows, Math.Max(1, (64 * 1024) / Math.Max(1, cols)), (s, e) =>
        {
            for (long row = s; row < e; row++) MaskRow((int)row);
        });
	}






		unsafe static public void replace_smaller(float* array, float* arrayIdx, int k, float data, float idx)
		{
			if (data < array[k - 1])
				return;
			for (int j = k - 2; j >= 0; j--)
			{
				if (data > array[j])
				{
					array[j + 1] = array[j];
					arrayIdx[j + 1] = arrayIdx[j];
				}
				else
				{
					array[j + 1] = data;
					arrayIdx[j + 1] = idx;
					return;
				}
			}
			array[0] = data;
			arrayIdx[0] = idx;
		}

		unsafe static public void TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k, int rows, int cols)
		{
			float* pOutVal = (float*)CpuNativeHelpers.GetBufferStart(outVal);
			float* pOutIdx = (float*)CpuNativeHelpers.GetBufferStart(outIdx);
			float* pInVal = (float*)CpuNativeHelpers.GetBufferStart(inVal);


			for (int j = 0; j < rows; ++j)
			{
				float* outputRow = pOutVal + j * k;
				float* outputIdxRow = pOutIdx + j * k;
				float* inputRow = pInVal + j * cols;

				for (int i = 0; i < k; ++i)
				{
					outputRow[i] = -1.70141e+38f;
					outputIdxRow[i] = -1.70141e+38f;

				}

				for (int i = 0; i < cols; i++)
				{
					replace_smaller(outputRow, outputIdxRow, k, inputRow[i], i);
				}
			}
		}

		unsafe static public void RoPE(Tensor tOut, Tensor tIn, int rows, int cols, int seqLen, int rowOffset)
		{
			float* result = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* src = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; ++j)
			{
				float* resultRow = result + j * cols;
				float* srcRow = src + j * cols;
				int m = (j % seqLen) + rowOffset;

				for (int id = 0; id < cols; id++)
				{
					int i = id / 2;
					float theta = (float)Math.Pow(500000.0, -2.0 * i / cols);
					float theta_m = theta * m;
					float cos_theta_m = (float)Math.Cos(theta_m);
					float sin_theta_m = (float)Math.Sin(theta_m);

					if (id % 2 == 0)
					{
						resultRow[id] = srcRow[id] * cos_theta_m - srcRow[id + 1] * sin_theta_m;
					}
					else
					{
						resultRow[id] = srcRow[id] * cos_theta_m + srcRow[id - 1] * sin_theta_m;
					}

				}
			}
		}

        unsafe static public void RoPEEx(Tensor tOut, Tensor tIn, Tensor positions, int rows, int cols, int ropeDim, int mode, float freqBase, float freqScale,
            int nCtxOrig = 0, float extFactor = 0.0f, float attnFactor = 1.0f, float betaFast = 0.0f, float betaSlow = 0.0f)
        {
            const int GGML_ROPE_TYPE_NEOX = 2;

            float* result = (float*)CpuNativeHelpers.GetBufferStart(tOut);
            float* src = (float*)CpuNativeHelpers.GetBufferStart(tIn);
            bool isNeoX = (mode & GGML_ROPE_TYPE_NEOX) != 0;
            int activeRopeDim = Math.Min(ropeDim, cols);
            int pairCount = activeRopeDim / 2;
            long rowBytes = (long)cols * sizeof(float);

            if (pairCount <= 0)
            {
                if (result != src)
                {
                    Buffer.MemoryCopy(src, result, rowBytes * rows, rowBytes * rows);
                }

                return;
            }

            bool useYarn = extFactor != 0.0f;

            float corrDimLow = 0, corrDimHigh = 0;
            float mscale = attnFactor;
            if (useYarn)
            {
                YarnCorrDims(activeRopeDim, nCtxOrig, freqBase, betaFast, betaSlow, out corrDimLow, out corrDimHigh);
                mscale *= 1.0f + 0.1f * MathF.Log(1.0f / freqScale);
            }

            float[] invFreqBuffer = ArrayPool<float>.Shared.Rent(pairCount);
            try
            {
                for (int i = 0; i < pairCount; i++)
                {
                    invFreqBuffer[i] = MathF.Pow(freqBase, -2.0f * i / activeRopeDim);
                }

                bool useIntPositions = positions.ElementType == DType.Int32;
                int* positionInts = useIntPositions ? (int*)CpuNativeHelpers.GetBufferStart(positions) : null;
                float* positionFloats = useIntPositions ? null : (float*)CpuNativeHelpers.GetBufferStart(positions);

                RoPEExRowsSimd(result, src, rows, cols, pairCount, isNeoX, useYarn, invFreqBuffer,
                    freqScale, corrDimLow, corrDimHigh, extFactor, mscale, positionInts, positionFloats);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(invFreqBuffer);
            }
        }

        /// <summary>
        /// RoPEEx rows on CpuParallel. Rows are usually laid out [seq, heads], so consecutive rows
        /// share a position: the cos/sin table is rebuilt only when the position changes (the
        /// previous loop paid 2 * pairCount trig calls per row, i.e. per head). The table values
        /// and the rotation arithmetic are exactly the scalar ones; the NeoX halves rotate as
        /// vectors, the interleaved layout stays scalar.
        /// </summary>
        unsafe static private void RoPEExRowsSimd(float* result, float* src, int rows, int cols, int pairCount, bool isNeoX,
            bool useYarn, float[] invFreq, float freqScale, float corrDimLow, float corrDimHigh, float extFactor, float mscale,
            int* positionInts, float* positionFloats)
        {
            long rowBytes = (long)cols * sizeof(float);
            CpuParallel.ForRange(rows, Math.Max(1, (8 * 1024) / Math.Max(1, cols)), [MethodImpl(MethodImplOptions.AggressiveOptimization)] (start, end) =>
            {
                float[] cosTable = new float[pairCount];
                float[] sinTable = new float[pairCount];
                int lastPosition = 0;
                bool haveTable = false;
                fixed (float* cosT = cosTable, sinT = sinTable)
                {
                    for (long row = start; row < end; row++)
                    {
                        float* resultRow = result + row * cols;
                        float* srcRow = src + row * cols;
                        if (resultRow != srcRow)
                        {
                            Buffer.MemoryCopy(srcRow, resultRow, rowBytes, rowBytes);
                        }

                        int position = positionInts != null ? positionInts[row] : (int)positionFloats[row];
                        if (!haveTable || position != lastPosition)
                        {
                            for (int i = 0; i < pairCount; i++)
                            {
                                float thetaExtrap = position * invFreq[i];
                                if (useYarn)
                                {
                                    YarnRoPE(thetaExtrap, freqScale, corrDimLow, corrDimHigh, i, extFactor, mscale, out cosT[i], out sinT[i]);
                                }
                                else
                                {
                                    float angle = thetaExtrap * freqScale;
                                    cosT[i] = MathF.Cos(angle);
                                    sinT[i] = MathF.Sin(angle);
                                }
                            }
                            lastPosition = position;
                            haveTable = true;
                        }

                        if (isNeoX)
                        {
                            int half = pairCount;
                            int i = 0;
                            if (Vector256.IsHardwareAccelerated)
                            {
                                for (; i + 8 <= half; i += 8)
                                {
                                    Vector256<float> left = Vector256.Load(srcRow + i);
                                    Vector256<float> right = Vector256.Load(srcRow + i + half);
                                    Vector256<float> c = Vector256.Load(cosT + i);
                                    Vector256<float> s = Vector256.Load(sinT + i);
                                    Vector256.Store(left * c - right * s, resultRow + i);
                                    Vector256.Store(right * c + left * s, resultRow + i + half);
                                }
                            }
                            for (; i < half; i++)
                            {
                                float left = srcRow[i];
                                float right = srcRow[i + half];
                                resultRow[i] = left * cosT[i] - right * sinT[i];
                                resultRow[i + half] = right * cosT[i] + left * sinT[i];
                            }
                        }
                        else
                        {
                            for (int i = 0, pair = 0; i < pairCount; ++i, pair += 2)
                            {
                                float left = srcRow[pair];
                                float right = srcRow[pair + 1];
                                resultRow[pair] = left * cosT[i] - right * sinT[i];
                                resultRow[pair + 1] = right * cosT[i] + left * sinT[i];
                            }
                        }
                    }
                }
            });
        }

        static private float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2.0f * MathF.PI)) / (2.0f * MathF.Log(freqBase));
        }

        static private void YarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow, out float low, out float high)
        {
            if (betaFast == 0.0f && betaSlow == 0.0f)
            {
                low = float.MaxValue;
                high = (float)(nDims / 2 - 1);
            }
            else
            {
                low = MathF.Max(0, MathF.Floor(YarnCorrDim(nDims, nCtxOrig, betaFast, freqBase)));
                high = MathF.Min(nDims / 2 - 1, MathF.Ceiling(YarnCorrDim(nDims, nCtxOrig, betaSlow, freqBase)));
            }
        }

        static private void YarnRoPE(float thetaExtrap, float freqScale, float corrDimLow, float corrDimHigh, int i0, float extFactor, float mscale,
            out float cosTheta, out float sinTheta)
        {
            float thetaInterp = freqScale * thetaExtrap;
            float rampY = ((float)i0 - corrDimLow) / MathF.Max(0.001f, corrDimHigh - corrDimLow);
            float rampMix = (1.0f - MathF.Min(1.0f, MathF.Max(0.0f, rampY))) * extFactor;
            float theta = thetaInterp * (1.0f - rampMix) + thetaExtrap * rampMix;
            cosTheta = MathF.Cos(theta) * mscale;
            sinTheta = MathF.Sin(theta) * mscale;
        }


		unsafe static public void RoPEGrad(Tensor tOut, Tensor tIn, int rows, int cols, int seqLen, int rowOffset)
		{
			float* grad = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; j++)
			{
				float* gradRow = grad + j * cols;
				float* adjRow = adj + j * cols;
				int m = (j % seqLen) + rowOffset;

				for (int id = 0; id < cols; id++)
				{
					int i = id / 2;
					float theta = (float)Math.Pow(500000.0, -2.0 * i / cols);
					float theta_m = theta * m;
					float cos_theta_m = (float)Math.Cos(theta_m);
					float sin_theta_m = (float)Math.Sin(theta_m);

					if (id % 2 == 0)
					{
						gradRow[id] += (adjRow[id] * cos_theta_m + adjRow[id + 1] * sin_theta_m);
					}
					else
					{
						gradRow[id] += (adjRow[id] * cos_theta_m - adjRow[id - 1] * sin_theta_m);
					}
				}
			}
		}

		unsafe static public void Softmax(Tensor tOut, Tensor tIn, int rows, int cols)
		{
			if (TryGetContiguousRows(tOut, out float* simdOut, out int simdOutRows, out int simdOutCols) &&
				TryGetContiguousRows(tIn, out float* simdIn, out int simdInRows, out int simdInCols) &&
				rows == simdOutRows && rows == simdInRows && cols == simdOutCols && cols == simdInCols)
			{
				CpuKernels.SoftmaxRows(simdOut, simdIn, rows, cols);
				return;
			}

			float* pOut = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* pIn = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; ++j)
			{
				float* so = pOut + j * cols;
				float* sp = pIn + j * cols;

				// Match GGML softmax: max starts at -inf so leading masked (-inf) logits behave like ggml_soft_max.
				float max = float.NegativeInfinity;
				for (int i = 0; i < cols; ++i)
				{
					max = Math.Max(max, sp[i]);
				}

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					float ex = (float)Math.Exp(sp[i] - max);
					so[i] = ex;
					sum += ex;
				}

				Span<float> spanSO = new Span<float>(so, cols);
				int vectorSize = Vector<float>.Count;
				int k = 0;
				Vector<float> vecSum = new Vector<float>(sum);
				for (k = 0; k < cols - vectorSize; k += vectorSize)
				{
					Vector<float> vecSO = new Vector<float>(spanSO.Slice(k));
					vecSO /= vecSum;

					vecSO.CopyTo(spanSO.Slice(k));
				}
				for (; k < cols; k++)
				{
					so[k] /= sum;
				}
			}
		}


		unsafe static public void SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_, int rows, int cols, bool addGrad)
		{

			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* val = (float*)CpuNativeHelpers.GetBufferStart(val_);

			for (int j = 0; j < rows; ++j)
			{
				float* gradRow = grad + j * cols;
				float* adjRow = adj + j * cols;
				float* valRow = val + j * cols;

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					sum += valRow[i] * adjRow[i];
				}

				for (int i = 0; i < cols; ++i)
				{
					if (addGrad)
					{
						gradRow[i] += valRow[i] * (adjRow[i] - sum);
					}
					else
					{
						gradRow[i] = valRow[i] * (adjRow[i] - sum);
					}
				}
			}
		}



		unsafe static public void IndexSelect(Tensor result_, Tensor src_, Tensor indice_, int rows, int cols, bool isAdd)
		{
            if (TryGetContiguousRows(result_, out float* contiguousResult, out int resultRows, out int resultCols) &&
                TryGetContiguousRows(src_, out float* contiguousSrc, out int srcRows, out int srcCols) &&
                resultRows == rows && resultCols == cols && srcCols == cols &&
                indice_.IsContiguous() && indice_.ElementCount() == rows)
            {
                void* contiguousIndice = (void*)CpuNativeHelpers.GetBufferStart(indice_);
                bool contiguousInt32 = indice_.ElementType == DType.Int32;
                long rowBytes = (long)cols * sizeof(float);

                void CopyRow(int row)
                {
                    int srcIdx = contiguousInt32 ? ((int*)contiguousIndice)[row] : (int)((float*)contiguousIndice)[row];
                    if (srcIdx < 0)
                    {
                        return;
                    }

                    if ((uint)srcIdx >= (uint)srcRows)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in index_select. Idx = '{srcIdx}', srcRows = '{srcRows}'");
                    }

                    float* resultRow = contiguousResult + row * cols;
                    float* srcRow = contiguousSrc + srcIdx * cols;
                    if (!isAdd)
                    {
                        Buffer.MemoryCopy(srcRow, resultRow, rowBytes, rowBytes);
                        return;
                    }

                    int vectorSize = Vector<float>.Count;
                    int i = 0;
                    for (; i <= cols - vectorSize; i += vectorSize)
                    {
                        StoreVec(resultRow + i, LoadVec(resultRow + i) + LoadVec(srcRow + i));
                    }

                    for (; i < cols; ++i)
                    {
                        resultRow[i] += srcRow[i];
                    }
                }

                CpuParallel.ForRange(rows, Math.Max(1, (64 * 1024) / Math.Max(1, cols)), (s, e) =>
                {
                    for (long row = s; row < e; row++) CopyRow((int)row);
                });

                return;
            }

			float* result = (float*)CpuNativeHelpers.GetBufferStart(result_);
			float* src = (float*)CpuNativeHelpers.GetBufferStart(src_);
			void* indice = (void*)CpuNativeHelpers.GetBufferStart(indice_);
			bool isInt32 = indice_.ElementType == DType.Int32;

			for (int j = 0; j < rows; j++)
			{
				int srcIdx = isInt32 ? ((int*)indice)[j] : (int)((float*)indice)[j];
				if (srcIdx >= 0)
				{
					float* resultRow = result + j * cols;
					float* srcRow = src + srcIdx * cols;

					for (int i = 0; i < cols; ++i)
					{
						if (isAdd == false)
						{
							resultRow[i] = srcRow[i];
						}
						else
						{
							resultRow[i] += srcRow[i];
						}
					}
				}
			}
		}


		unsafe static public void IndexSelectGrad(Tensor grad_, Tensor adj_, Tensor indice_, int rows, int cols)
		{
			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* indice = (float*)CpuNativeHelpers.GetBufferStart(indice_);

			for (int j = 0; j < rows; j++)
			{
				int gradIdx = (int)indice[j];
				if (gradIdx >= 0)
				{
					float* adjRow = adj + j * cols;
					float* gradRow = grad + gradIdx * cols;

					for (int i = 0; i < cols; ++i)
					{
						gradRow[i] += adjRow[i];
					}
				}
			}
		}



		unsafe static public void LayerNorm(Tensor out_,
			Tensor in_,
			Tensor gamma_,
			Tensor beta_,
			float eps,
			int rows,
			int cols)
		{
			if (gamma_ != null &&
				TryGetContiguousRows(out_, out float* simdOut, out int simdOutRows, out int simdOutCols) &&
				TryGetContiguousRows(in_, out float* simdIn, out int simdInRows, out int simdInCols) &&
				TryGetContiguousFloat(gamma_, out float* simdGamma, out int simdGammaLength) &&
				rows == simdOutRows && rows == simdInRows && cols == simdOutCols && cols == simdInCols &&
				simdGammaLength == cols)
			{
				float* simdBeta = null;
				if (beta_ == null || (TryGetContiguousFloat(beta_, out simdBeta, out int simdBetaLength) && simdBetaLength == cols))
				{
					CpuKernels.LayerNormRows(simdOut, simdIn, simdGamma, simdBeta, rows, cols, eps);
					return;
				}
			}

			float* outPtr = (float*)CpuNativeHelpers.GetBufferStart(out_);
			float* inPtr = (float*)CpuNativeHelpers.GetBufferStart(in_);
			float* alpha = (float*)CpuNativeHelpers.GetBufferStart(gamma_!);
			float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;

			for (int j = 0; j < rows; ++j)
			{
				float* so = outPtr + j * cols;
				float* sp = inPtr + j * cols;

				Span<float> spanSP = new Span<float>(sp, cols);

				float sum = 0.0f;
				int vectorSize = Vector<float>.Count;
				Vector<float> vecAdded = Vector<float>.Zero;
				int i = 0;

				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					vecAdded += vecSp;
				}
				sum = Vector.Dot(vecAdded, Vector<float>.One);
				for (; i < cols; i++)
				{
					sum += sp[i];
				}

				float mean = sum / cols;
				float sqSum = 0.0f;

				Vector<float> vecMean = new Vector<float>(mean);
				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecEx = vecSp - vecMean;
					sqSum += Vector.Dot(vecEx, vecEx);
				}
				for (; i < cols; ++i)
				{
					float ex = sp[i] - mean;
					sqSum += ex * ex;
				}

				float sigma = (float)Math.Sqrt(eps + sqSum / cols);

				Span<float> spanSO = new Span<float>(so, cols);
				Span<float> spanAlpha = new Span<float>(alpha, cols);
				bool hasBeta = beta != null;
				Span<float> spanBeta = hasBeta ? new Span<float>(beta, cols) : Span<float>.Empty;
				Vector<float> vecSigma = new Vector<float>(sigma);

				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecAlpha = new Vector<float>(spanAlpha.Slice(i));

					Vector<float> vecT = vecAlpha * ((vecSp - vecMean) / vecSigma);

					if (hasBeta)
					{
						Vector<float> vecBeta = new Vector<float>(spanBeta.Slice(i));
						vecT += vecBeta;
					}

					vecT.CopyTo(spanSO.Slice(i));
				}
				for (; i < cols; ++i)
				{
					float t = alpha[i] * ((sp[i] - mean) / sigma);
					if (beta != null)
					{
						t += beta[i];
					}

					so[i] = t;
				}


			}
		}



		unsafe static public void RMSNorm(Tensor out_,
			Tensor in_,
			Tensor gamma_,
			Tensor beta_,
			float eps,
			int rows,
			int cols)
		{
            // gamma_ == null means an unweighted RMS norm (x * rsqrt(mean(x^2)+eps)),
            // which several architectures use for per-head query/key normalization.
            float* gammaPtr = null;
            int gammaLength = cols;
            bool gammaOk = gamma_ == null || TryGetContiguousFloat(gamma_, out gammaPtr, out gammaLength);
            if (gammaOk &&
                TryGetContiguousRows(out_, out float* contiguousOut, out int outRows, out int outCols) &&
                TryGetContiguousRows(in_, out float* contiguousIn, out int inRows, out int inCols) &&
                rows == outRows && rows == inRows && cols == outCols && cols == inCols && gammaLength == cols &&
                (beta_ == null || (TryGetContiguousFloat(beta_, out _, out int betaLength) && betaLength == cols)))
            {
                float* betaPtr = beta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;
                bool hasGamma = gammaPtr != null;
                // Without a gain the bias is not applied.
                CpuKernels.RmsNormRows(contiguousOut, contiguousIn, gammaPtr, hasGamma ? betaPtr : null, rows, cols, eps);
                return;
            }

			float* outPtr = (float*)CpuNativeHelpers.GetBufferStart(out_);
			float* inPtr = (float*)CpuNativeHelpers.GetBufferStart(in_);
			float* gamma = (gamma_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(gamma_) : null;
			float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;
			bool bias = (beta_ != null);

            float N = cols;
			for (int j = 0; j < rows; j++)
			{
				float* yRow = outPtr + j * cols;
				float* xRow = inPtr + j * cols;

				float _sqSum = 0;

				for (int id = 0; id < cols; id++)
				{
					float xv = (float)xRow[id];

					_sqSum += xv * xv;

				}

				float rms = (float)Math.Sqrt(_sqSum / N + eps); 
				for (int id = 0; id < cols; id++)
				{

					float gammav = gamma != null ? gamma[id] : 1.0f;
					float xv = xRow[id];
					float betav = bias ? beta[id] : 0.0f;
					float rmsNorm = xv / rms;
					float y = gammav * rmsNorm + betav;
					yRow[id] = y;

				}
			}


		}



        unsafe static public void Adam(Tensor tw, Tensor tg, Tensor tv, Tensor tm, int rows, int cols, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
		{
			float* w = (float*)CpuNativeHelpers.GetBufferStart(tw);
			float* g = (float*)CpuNativeHelpers.GetBufferStart(tg);
			float* v = (float*)CpuNativeHelpers.GetBufferStart(tv);
			float* m = (float*)CpuNativeHelpers.GetBufferStart(tm);

			for (int j = 0; j < rows; j++)
			{
				float* sw = w + j * cols;
				float* sg = g + j * cols;
				float* sv = v + j * cols;
				float* sm = m + j * cols;

				for (int i = 0; i < cols; i++)
				{
					if (sg[i] != 0.0)
					{
						float g2 = sg[i] * gradNormFactor;

						if (g2 > clipval)
						{
							g2 = clipval;
						}
						if (g2 < -clipval)
						{
							g2 = -clipval;
						}

						sm[i] = sm[i] * decay_rate_m + (1.0f - decay_rate_m) * g2;
						sv[i] = sv[i] * decay_rate_v + (1.0f - decay_rate_v) * g2 * g2;

						double m_cap = sm[i] / (1.0 - Math.Pow(decay_rate_m, iter));
						double v_cap = sv[i] / (1.0 - Math.Pow(decay_rate_v, iter));

						sw[i] -= (float)(step_size * m_cap / (Math.Sqrt(v_cap) + eps));

						sg[i] = 0;
					}
				}
			}
		}


		#region Internal operations


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relu(float w)
		{
			if (w < 0.0f)
				return 0.0f;
			return w;

		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relud(float w, float g)
		{
			if (w > 0.0f)
				return g;
			return 0.0f;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addrelud(float t, float w, float g)
		{
			if (w > 0.0f)
				return t + g;
			return t;
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyReLU(float w)
        {
            if (w < 0.0f)
                return 0.01f * w;
            return w;

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyReLUD(float w, float g)
        {
            if (w >= 0.0f)
                return g;
            return 0.01f * g;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float AddLeakyReLUD(float t, float w, float g)
        {
            if (w >= 0.0f)
                return t + g;
            return t + 0.01f * g;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float SiLU(float w)
		{
			return w / (1.0f + (float)Math.Exp(-w));
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float GELU(float x)
		{
			return 0.5f * x * (1.0f + (float)Math.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float SiLUD(float w, float resG)
		{
			float sig = 1.0f / (1.0f + (float)Math.Exp(-w));
			float grad = sig * (1.0f + w * (1.0f - sig));
			return resG * grad;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float AddSiLUD(float t, float w, float resG)
		{
			float sig = 1.0f / (1.0f + (float)Math.Exp(-w));
			float grad = sig * (1.0f + w * (1.0f - sig));
			return t + resG * grad;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float add(float x, float y)
		{
			return x + y;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float mul(float x, float y)
		{
			return x * y;
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float div(float x, float y)
		{
			return x / y;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float sigmoid(float x)
		{
			return (float)(1.0 / (1.0 + Math.Exp(-x)));
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float sigmoidD(float resW, float resG)
		{
			return resW * (1.0f - resW) * resG;
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addSigmoidD(float t, float resW, float resG)
		{
			return t + resW * (1.0f - resW) * resG;
		}



		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float mulmuladd(float x, float y, float z, float w)
		{
			return x * y + z * w;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float clamp(float val, float min, float max)
		{
			if (val < min)
				return min;
			if (val > max)
				return max;
			return val;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addmul(float x, float y, float z)
		{
			return x + y * z;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float adddiv(float x, float y, float z)
		{
			return x + y / z;
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addtanh(float x, float y)
		{
			return (float)Math.Tanh(x + y);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addtanhD(float t, float resW, float resG)
		{
			return t + (1.0f - resW * resW) * resG;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float tanhD(float resW, float resG)
		{
			return (1.0f - resW * resW) * resG;
		}

		#endregion
	}
}
