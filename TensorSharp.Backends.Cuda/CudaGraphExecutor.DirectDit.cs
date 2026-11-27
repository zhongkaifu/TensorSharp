// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The CUDA executor's direct products (TensorGraphExecutor.DirectProducts) and the regions a diffusion transformer that ran on
// the direct CUDA engines takes their kernels for (LTX 2.5's denoiser): per-head rotary tables, per-head output gates and F32
// attention. Each serves its region with the kernel the direct engine ran, so a program that replaces such an engine computes
// what it computed.
using System;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    public sealed partial class CudaGraphExecutor
    {
        // The model-owned kernels the direct engines kept a session of: the K-quant projection module (the F16 prefill route
        // and the small-row F32 tiles) and the bounded F32 attention and head-gate kernels. Made on first use.
        private CudaLtx25ProjectionSession _directProjections;
        private CudaLtx25AttentionSession _directAttention;

        private CudaLtx25ProjectionSession DirectProjections =>
            _directProjections ??= new CudaLtx25ProjectionSession((CudaAllocator)Allocator);

        private CudaLtx25AttentionSession DirectAttention =>
            _directAttention ??= new CudaLtx25AttentionSession((CudaAllocator)Allocator);

        /// <summary>Direct products and served attentions that ran (tests).</summary>
        internal long DirectProductsForTests, FullAttentionsServedForTests, HeadRotariesServedForTests, HeadGatesServedForTests;

        /// <summary>
        /// A product as DirectLinear runs it on CUDA (<see cref="TensorGraphExecutor.DirectProducts"/>): an exact F32 weight on the
        /// pedantic F32 GEMM (<see cref="CudaLtx25LoraOps.TryLinear"/>); a Q4_K, Q5_K or Q6_K weight over a prompt's rows on the
        /// F16 GEMM over its F16 dequantization, over 2-31 rows on the F32 tile kernels; every other product the resident one
        /// (<see cref="CudaQuantizedOps.TryAddmmQuantizedToFloat32"/>). A product none of them takes is an error: running another
        /// route would change the numbers without a word.
        /// </summary>
        private Tensor DirectProduct(IGraphWeight weight, Tensor x)
        {
            if (weight.Ne2 > 1 || weight.Data == IntPtr.Zero || x.Storage is not CudaStorage || x.ElementType != DType.Float32
                || !x.IsContiguous() || x.DimensionCount != 2)
                throw new NotSupportedException($"a direct product takes a 2-D weight over contiguous F32 rows ([{weight.Ne1} x {weight.Ne0}]).");
            var result = new Tensor(Allocator, DType.Float32, x.Sizes[0], weight.Ne1);
            try
            {
                bool done = weight.ExactProducts
                    ? weight.Format == GraphFormat.F32 && CudaLtx25LoraOps.TryLinear(result, x, weight.Data, weight.Ne0, weight.Ne1, weight.Bytes)
                    : DirectProjections.TryPrefillLinear(result, x, weight.Data, weight.Format, weight.Ne0, weight.Ne1, weight.Bytes)
                      || DirectProjections.TryLinear(result, x, weight.Data, weight.Format, weight.Ne0, weight.Ne1, weight.Bytes)
                      || CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, x, weight.CacheKey, weight.Data, weight.Format,
                          weight.Ne0, weight.Ne1, weight.Bytes);
                if (!done)
                    throw new InvalidOperationException(
                        $"CUDA direct product failed (type {weight.Format}, [{x.Sizes[0]},{weight.Ne0}]x[{weight.Ne0},{weight.Ne1}]).");
                DirectProductsForTests++;
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        /// <see cref="GraphFusion.FullAttention"/> as the direct CUDA engines attended: a sequence whose queries and keys both
        /// number 512 or more (or past 100 GFLOP) on the pedantic F32 batched products over a bounded score workspace (every
        /// query row against all keys, <see cref="CudaLtx25AttentionSession.TryAttention"/>), a shorter one on the streaming F32
        /// kernel (<see cref="CudaWanOps.TryAttention"/>), each the other's fallback; then its head gates in place, 2 * sigmoid on
        /// the fused kernel (<see cref="CudaLtx25AttentionSession.TryApplyHeadGates"/>), else as separate ops. False (the region's
        /// ops run) where neither attention takes the layout.
        /// </summary>
        private bool TryFullAttention(Tensor[] inputs, float[] fp, Tensor[] results)
        {
            if (inputs.Length is not (3 or 4) || fp == null || fp.Length < (inputs.Length == 4 ? 2 : 1) || results.Length < 1) return false;
            using Tensor q = Rows3(inputs[0]), k = Rows3(inputs[1]), v = Rows3(inputs[2]);
            if (q == null || k == null || v == null || !q.IsContiguous() || !k.IsContiguous() || !v.IsContiguous()
                || q.ElementType != DType.Float32 || k.ElementType != DType.Float32 || v.ElementType != DType.Float32
                || q.Storage is not CudaStorage || k.Storage is not CudaStorage || v.Storage is not CudaStorage)
                return false;
            long sq = q.Sizes[0], sk = k.Sizes[0], heads = q.Sizes[1], hd = q.Sizes[2];
            if (k.Sizes[1] != heads || v.Sizes[1] != heads || k.Sizes[2] != hd || v.Sizes[2] != hd || v.Sizes[0] != sk
                || sq > int.MaxValue || sk > int.MaxValue || heads > 128 || hd > 1024)
                return false;
            Tensor logits = inputs.Length == 4 ? inputs[3] : null;
            if (logits != null && (logits.ElementType != DType.Float32 || !logits.IsContiguous() || logits.ElementCount() != sq * heads))
                return false;
            float scale = fp[0];
            using Tensor q2 = q.View(sq, heads * hd), k2 = k.View(sk, heads * hd), v2 = v.View(sk, heads * hd);
            var output = new Tensor(Allocator, DType.Float32, sq, heads * hd);
            try
            {
                bool batched = (sq >= 512 && sk >= 512) || 4.0 * sq * sk * heads * hd > 100e9;
                bool done = batched
                    ? DirectAttention.TryAttention(output, q2, k2, v2, (int)heads, (int)hd, scale)
                      || CudaWanOps.TryAttention(output, q2, k2, v2, null, (int)heads, (int)sq, (int)sk, (int)hd, scale)
                    : CudaWanOps.TryAttention(output, q2, k2, v2, null, (int)heads, (int)sq, (int)sk, (int)hd, scale)
                      || DirectAttention.TryAttention(output, q2, k2, v2, (int)heads, (int)hd, scale);
                if (!done)
                {
                    output.Dispose();
                    return false;
                }
                if (logits != null)
                {
                    using Tensor l2 = logits.View(sq, heads);
                    if (fp[1] == 2f && DirectAttention.TryApplyHeadGates(output, l2, (int)heads, (int)hd)) HeadGatesServedForTests++;
                    else GateHeads(output, l2, (int)heads, fp[1], contiguousGates: true);
                }
            }
            catch
            {
                output.Dispose();
                throw;
            }
            results[0] = output;
            FullAttentionsServedForTests++;
            return true;
        }

        /// <summary>
        /// <see cref="GraphFusion.HeadRotary"/> on the fused NeoX rotary kernel the direct CUDA engines rotated per-head tables
        /// with (<see cref="CudaFusedOps.TryNeoXRoPEFlatInPlace"/>: each token and head a row of its own table), in place on the
        /// RMS norm's result where the region norms first (the direct engines' order), else on a copy of x.
        /// </summary>
        private bool TryHeadRotary(Tensor[] inputs, long[] ip, float[] fp, Tensor[] results)
        {
            bool normed = ip is { Length: > 0 } && ip[0] == 1;
            if (inputs.Length != (normed ? 4 : 3) || results.Length < 1 || (normed && (fp == null || fp.Length < 1))) return false;
            Tensor x = inputs[0], cos = inputs[1], sin = inputs[2], weight = normed ? inputs[3] : null;
            if (x == null || cos == null || sin == null || x.Storage is not CudaStorage || x.ElementType != DType.Float32 || !x.IsContiguous()
                || cos.ElementType != DType.Float32 || sin.ElementType != DType.Float32 || !cos.IsContiguous() || !sin.IsContiguous())
                return false;
            long tokens = x.Sizes[0], width = x.ElementCount() / Math.Max(1, tokens);
            long half = cos.ElementCount() / Math.Max(1, tokens);
            long heads = cos.DimensionCount >= 2 ? cos.Sizes[^2] : 0, d = 2 * half / Math.Max(1, heads);
            if (tokens <= 0 || heads <= 0 || d % 2 != 0 || heads * d != width || sin.ElementCount() != cos.ElementCount()
                || tokens * heads > int.MaxValue || (weight != null && (weight.ElementType != DType.Float32 || weight.ElementCount() != width)))
                return false;
            using Tensor x2 = x.View(tokens, width);
            Tensor y;
            if (weight != null)
            {
                using Tensor w = weight.View(width);
                y = Ops.RMSNorm(null, x2, w, null, fp[0]);
            }
            else y = Ops.NewContiguous(x2);
            try
            {
                long rows = tokens * heads, rowHalf = d / 2;
                using Tensor flat = y.View(rows, d);
                using Tensor c = cos.View(rows * rowHalf), s = sin.View(rows * rowHalf);
                if (!CudaFusedOps.TryNeoXRoPEFlatInPlace(flat, c, s, 1, (int)rows, (int)d, (int)rowHalf))
                {
                    y.Dispose();
                    return false;
                }
                results[0] = y.View(tokens, heads, d);
            }
            finally
            {
                y.Dispose();
            }
            HeadRotariesServedForTests++;
            return true;
        }

        /// <summary>Return the direct kernels' scratch (the F16 weight and activation buffers of the prefill route, the attention's
        /// score workspace) to the device: a family calls it when its denoising ends, as the direct engines released theirs
        /// before the decode.</summary>
        public void ReleaseDirectWorkspaces()
        {
            _directProjections?.ReleaseWorkspace();
            _directAttention?.ReleaseWorkspace();
        }

        private void DisposeDirect()
        {
            _directProjections?.Dispose();
            _directProjections = null;
            _directAttention?.Dispose();
            _directAttention = null;
        }
    }
}
