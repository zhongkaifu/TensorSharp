// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;

namespace TensorSharp.MLX
{
    /// <summary>
    /// Products with unquantized F32, F16 and BF16 weights kept as their GGUF bytes on the device: a matrix-vector kernel for
    /// decode-sized row counts (each weight element read once, products and sums in F32, F32 out, as ggml-metal's
    /// kernel_mul_mv_f16_f32 computes them), and MLX's GEMM above that (F32 weights as they are; F16 in half with F32 sums and
    /// the result widened, as the Q6_K and IQ4_XS prompt products run; BF16 widened to exact F32 in bounded slices).
    /// </summary>
    internal static partial class MlxNative
    {
        /// <summary>Input rows up to which an F32/F16/BF16 weight runs as the matrix-vector kernel (one read of the weight for
        /// every row). Settable for tests.</summary>
        internal static int FloatMatvecMaxRows = 8;

        private static readonly Dictionary<(int Type, int Rows), MlxFastMetalKernel> floatMatvecKernels = new();

        // One threadgroup of two simdgroups covers 8 output rows (4 a simdgroup) for all R input rows: lanes stride the
        // input dimension by 4 elements (one 4-wide load of each of the simdgroup's weight rows and of every input row),
        // sums stay in F32 registers, and one simd_sum per (output row, input row) finishes them. A 4-wide tail-free loop
        // needs in_dim % 4 == 0 (the caller checks).
        private static string FloatMatvecSource(int ggmlType, int rows) => $$"""
            constexpr int R = {{rows}};
            constexpr int NR = 4;
            const int in_dim4 = in_dim / 4;
            const int first = (int(threadgroup_position_in_grid.x) * 2 + int(simdgroup_index_in_threadgroup)) * NR;
            const ushort lane = thread_index_in_simdgroup;
            if (first >= out_dim) {
                return;
            }
            const int nrows = min(NR, out_dim - first);
            float sums[NR][R];
            for (int o = 0; o < NR; ++o) for (int r = 0; r < R; ++r) sums[o][r] = 0.f;
            device const float4 * x4 = reinterpret_cast<device const float4 *>(x);
            for (int i = lane; i < in_dim4; i += 32) {
                float4 xv[R];
                for (int r = 0; r < R; ++r) xv[r] = x4[uint64_t(r) * uint64_t(in_dim4) + uint64_t(i)];
                for (int o = 0; o < NR; ++o) {
                    if (o < nrows) {
                        float4 wv = {{FloatMatvecLoad(ggmlType)}};
                        for (int r = 0; r < R; ++r) sums[o][r] += dot(wv, xv[r]);
                    }
                }
            }
            for (int o = 0; o < nrows; ++o) {
                for (int r = 0; r < R; ++r) {
                    float total = simd_sum(sums[o][r]);
                    if (lane == 0) {
                        y[uint64_t(r) * uint64_t(out_dim) + uint64_t(first + o)] = total;
                    }
                }
            }
            """;

        private static string FloatMatvecLoad(int ggmlType) => ggmlType switch
        {
            (int)GgmlTensorType.F32 =>
                "reinterpret_cast<device const float4 *>(w)[uint64_t(first + o) * uint64_t(in_dim4) + uint64_t(i)]",
            (int)GgmlTensorType.F16 =>
                "float4(reinterpret_cast<device const half4 *>(w)[uint64_t(first + o) * uint64_t(in_dim4) + uint64_t(i)])",
            (int)GgmlTensorType.BF16 =>
                "tensorsharp_bf16x4(reinterpret_cast<device const ushort4 *>(w)[uint64_t(first + o) * uint64_t(in_dim4) + uint64_t(i)])",
            _ => throw new NotSupportedException($"No float matrix-vector load for GGML type {(GgmlTensorType)ggmlType}."),
        };

        private const string FloatMatvecHeader = """
            inline float4 tensorsharp_bf16x4(ushort4 b) {
                return float4(as_type<float>(uint(b.x) << 16), as_type<float>(uint(b.y) << 16),
                              as_type<float>(uint(b.z) << 16), as_type<float>(uint(b.w) << 16));
            }
            """;

        /// <summary>Whether <see cref="FloatWeightMatmul"/> takes this GGML type.</summary>
        internal static bool IsFloatWeightType(int ggmlType)
            => ggmlType is (int)GgmlTensorType.F32 or (int)GgmlTensorType.F16 or (int)GgmlTensorType.BF16;

        private static MlxFastMetalKernel EnsureFloatMatvecKernel(int ggmlType, int rows)
        {
            lock (fastKernelSync)
            {
                if (floatMatvecKernels.TryGetValue((ggmlType, rows), out MlxFastMetalKernel kernel) && kernel.IsValid)
                    return kernel;
                string label = ((GgmlTensorType)ggmlType).ToString().ToLowerInvariant();
                kernel = CreateFastMetalKernel($"tensorsharp_{label}_matvec_r{rows}", new[] { "x", "w", "in_dim", "out_dim" }, new[] { "y" },
                    FloatMatvecSource(ggmlType, rows), FloatMatvecHeader);
                if (!kernel.IsValid)
                    throw new NotSupportedException($"Unable to initialize the MLX {(GgmlTensorType)ggmlType} matrix-vector kernel.");
                floatMatvecKernels[(ggmlType, rows)] = kernel;
                return kernel;
            }
        }

        /// <summary>
        /// x [rows, inDim] (F32) times a raw F32/F16/BF16 weight [outDim, inDim] (its GGUF bytes, any MLX array of them),
        /// transposed: [rows, outDim] F32. Runs on the MLX worker.
        /// </summary>
        internal static MlxArray FloatWeightMatmul(MlxArray input, MlxArray rawWeight, int ggmlType, int rows, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("A float-weight product needs valid input and weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows), "A float-weight product needs positive dimensions.");
            return MlxWorker.Shared.Invoke(() => rows <= FloatMatvecMaxRows && inDim % 4 == 0
                ? FloatMatvec(input, rawWeight, ggmlType, rows, inDim, outDim)
                : FloatGemm(input, rawWeight, ggmlType, rows, inDim, outDim));
        }

        private static MlxArray FloatMatvec(MlxArray input, MlxArray rawWeight, int ggmlType, int rows, int inDim, int outDim)
        {
            MlxFastMetalKernel kernel = EnsureFloatMatvecKernel(ggmlType, rows);
            string label = ((GgmlTensorType)ggmlType).ToString();
            MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
            MlxVectorArray inputs = default, outputs = default;
            MlxArray inputF32 = default, contiguous = default, inDimArray = default, outDimArray = default;
            try
            {
                int[] shape = { rows, outDim };
                Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} matrix-vector output");
                Check(mlx_fast_metal_kernel_config_set_grid(config, 64 * ((outDim + 7) / 8), 1, 1), $"configuring {label} matrix-vector grid");
                Check(mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1), $"configuring {label} matrix-vector threadgroup");
                Check(mlx_astype(out inputF32, input, ToMlxDtype(DType.Float32), DefaultStream()), $"casting {label} matrix-vector input");
                Check(mlx_contiguous(out contiguous, inputF32, false, DefaultStream()), $"laying out {label} matrix-vector input");
                inDimArray = mlx_array_new_int(inDim);
                outDimArray = mlx_array_new_int(outDim);
                inputs = CreateVectorArray(contiguous, rawWeight, inDimArray, outDimArray);
                outputs = mlx_vector_array_new();
                Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} matrix-vector product");
                Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} matrix-vector output");
                return result;
            }
            finally
            {
                FreeArray(inputF32); FreeArray(contiguous); FreeArray(inDimArray); FreeArray(outDimArray);
                if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
            }
        }

        /// <summary>
        /// MLX's GEMM over the weight's own bytes: F32 as it is; F16 in half (the input rounded to half, F32 sums, the product
        /// widened back: the precision of ggml-metal's mul_mm and of the Q6_K prompt products); BF16 widened to exact F32 a
        /// bounded slice of output rows at a time (bf16 products would round the result to 8 bits).
        /// </summary>
        private static MlxArray FloatGemm(MlxArray input, MlxArray rawWeight, int ggmlType, int rows, int inDim, int outDim)
        {
            string label = ((GgmlTensorType)ggmlType).ToString();
            // mlx_dtype: float16 9, float32 10, bfloat16 12.
            int typed = ggmlType switch
            {
                (int)GgmlTensorType.F32 => 10,
                (int)GgmlTensorType.F16 => 9,
                _ => 12,
            };
            int compute = ggmlType == (int)GgmlTensorType.F16 ? 9 : 10;
            // Slices of output rows bounded at 256 MiB of widened weight (a 128K-vocabulary head is 1 GB in F32).
            int sliceRows = (int)Math.Clamp(DequantSliceBytes / (4L * inDim), 1, outDim);
            var parts = new List<MlxArray>();
            MlxArray x = default, all = default, product = default, joined = default;
            MlxVectorArray partVector = default;
            try
            {
                Check(mlx_astype(out x, input, compute, DefaultStream()), $"casting {label} matmul input");
                all = ViewRawAs(rawWeight, typed, outDim, inDim, label);
                for (int row0 = 0; row0 < outDim; row0 += sliceRows)
                {
                    int count = Math.Min(sliceRows, outDim - row0);
                    MlxArray slice = default, widened = default, weightT = default, part = default;
                    bool sliced = count != outDim;
                    try
                    {
                        slice = sliced ? SliceRows(all, row0, count, inDim) : all;
                        MlxArray source = slice;
                        if (typed != compute)
                        {
                            Check(mlx_astype(out widened, slice, compute, DefaultStream()), $"widening {label} weight");
                            source = widened;
                        }
                        Check(mlx_transpose_axes(out weightT, source, new[] { 1, 0 }, 2, DefaultStream()), $"transposing {label} weight");
                        Check(mlx_matmul(out part, x, weightT, DefaultStream()), $"running {label} matmul");
                        // Several slices: each finished before the next widened copy exists.
                        if (sliceRows < outDim) Eval(part);
                        parts.Add(part);
                        part = default;
                    }
                    finally
                    {
                        FreeArray(part); FreeArray(weightT); FreeArray(widened);
                        if (sliced) FreeArray(slice);
                    }
                }
                if (parts.Count == 1)
                {
                    product = parts[0];
                    parts.Clear();
                }
                else
                {
                    partVector = CreateVectorArray(parts.ToArray());
                    Check(mlx_concatenate_axis(out product, partVector, 1, DefaultStream()), $"joining {label} matmul slices");
                }
                if (compute == 10)
                {
                    MlxArray result = product;
                    product = default;
                    return result;
                }
                Check(mlx_astype(out joined, product, 10, DefaultStream()), $"casting {label} matmul output");
                MlxArray output = joined;
                joined = default;
                return output;
            }
            finally
            {
                foreach (MlxArray part in parts) FreeArray(part);
                if (partVector.IsValid) _ = mlx_vector_array_free(partVector);
                FreeArray(x); FreeArray(all); FreeArray(product); FreeArray(joined);
            }
        }

        [LibraryImport(LibraryName, EntryPoint = "mlx_view")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_view(out MlxArray result, MlxArray array, int dtype, MlxStream stream);

        /// <summary>The raw bytes of a [rows, cols] weight as an MLX array of mlx dtype <paramref name="type"/> (a view: no
        /// copy).</summary>
        private static MlxArray ViewRawAs(MlxArray raw, int type, int rows, int cols, string label)
        {
            MlxArray typed = default, shaped = default;
            try
            {
                Check(mlx_view(out typed, raw, type, DefaultStream()), $"viewing {label} weight bytes");
                Check(mlx_reshape(out shaped, typed, new[] { rows, cols }, 2, DefaultStream()), $"shaping {label} weight");
                MlxArray result = shaped;
                shaped = default;
                return result;
            }
            finally
            {
                FreeArray(typed);
                FreeArray(shaped);
            }
        }

        private static MlxArray SliceRows(MlxArray array, int row0, int rows, int cols)
        {
            Check(mlx_slice(out MlxArray slice, array, new[] { row0, 0 }, 2, new[] { row0 + rows, cols }, 2, new[] { 1, 1 }, 2, DefaultStream()),
                "slicing weight rows");
            return slice;
        }
    }
}
