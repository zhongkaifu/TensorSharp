// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The direct CUDA backend's executor plugin: programs run node by node on CUDA tensors, with the
    /// backend's quantized products (weights resident under their device cache keys).
    /// </summary>
    public sealed class CudaGraphExecutor : TensorGraphExecutor
    {
        [ModuleInitializer]
        internal static void Register()
            => GraphExecutors.Register(typeof(CudaAllocator), allocator => new CudaGraphExecutor(allocator));

        public CudaGraphExecutor(IAllocator allocator) : base(allocator) { }

        /// <summary>CUDA's binary kernels take same-shaped contiguous operands: a broadcast operand is gathered.</summary>
        protected override bool BroadcastsInKernels => false;

        /// <summary>A prompt chunk's or a decode row's K/V rows into the head-first cache (F32 or F16) with the
        /// per-op path's kernel.</summary>
        protected override bool TryWriteRows(Tensor dst, Tensor src, int first, int count)
        {
            if (dst.DimensionCount != 3 || !dst.IsContiguous()) return false;
            using Tensor rows = src.View(src.Sizes[^3], src.Sizes[^2], src.Sizes[^1]);
            Tensor contiguous = rows.IsContiguous() ? rows.CopyRef() : Ops.NewContiguous(rows);
            using (contiguous)
                return CudaFusedOps.TryCopyHeadFirstToCache(dst, contiguous, first, count, (int)dst.Sizes[1], false);
        }

        /// <summary>
        /// Attention over cache windows with the per-op path's kernels: a decode row runs the GQA decode kernel
        /// on the head-first cache in place (F32 or F16); a prompt chunk expands the window's KV heads, runs the
        /// half-operand GEMMs of the per-op prefill with the causal mask, and the softmax. Anything else (an
        /// explicit mask, an operand that is not a cache window) takes the generic path.
        /// </summary>
        protected override Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            if (kWindow == null || vWindow == null || mask.Pattern == MaskPattern.Explicit
                || k.Sizes[2] != v.Sizes[2] || q.ElementType != DType.Float32)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2];
            int kvHeads = (int)k.Sizes[0], len = (int)k.Sizes[1], floor = mask.Floor;
            Tensor kc = kWindow.Cache, vc = vWindow.Cache;
            int cap = (int)kc.Sizes[1];
            using Tensor qc = q.IsContiguous() ? q.CopyRef() : Ops.NewContiguous(q);
            if (nq == 1)
            {
                using Tensor query = qc.View(1, heads * d);
                var result = new Tensor(Allocator, DType.Float32, 1, heads * d);
                if (CudaFusedOps.TryGqaDecodeAttention(result, query, kc, vc, heads, kvHeads, d,
                        (int)kWindow.Start + floor, len - floor, cap, false, scale))
                {
                    using (result) return result.View(1, heads, d);
                }
                result.Dispose();
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            }
            if (kWindow.Start != 0 || vWindow.Start != 0 || heads % kvHeads != 0)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            int group = heads / kvHeads;
            // K and V once as F32 [kvHeads, len, d] (the cache may hold F16); the query heads of a group share
            // their KV head, so q [heads, rows, d] is [kvHeads, group * rows, d] and each product is one batch
            // per KV head (no per-head copies of the cache).
            using var kx = new Tensor(Allocator, DType.Float32, kvHeads, len, d);
            using var vx = new Tensor(Allocator, DType.Float32, kvHeads, len, d);
            if (!CudaFusedOps.TryExpandKvHeads(kx, kc, 1, len) || !CudaFusedOps.TryExpandKvHeads(vx, vc, 1, len))
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            using Tensor kt = kx.Transpose(1, 2);
            var output = new Tensor(Allocator, DType.Float32, nq, heads, d);
            // Query tiles keep the scores [heads, tile, len] within ScoreBudgetBytes at any context length.
            int tile = (int)Math.Clamp(ScoreBudgetBytes / ((long)heads * len * sizeof(float)), 1, nq);
            for (int first = 0; first < nq; first += tile)
            {
                int rows = Math.Min(tile, nq - first);
                using Tensor qRows = qc.Narrow(1, first, rows);
                using Tensor qTile = rows == nq ? qc.CopyRef() : Ops.NewContiguous(qRows);
                using Tensor q3 = qTile.View(kvHeads, (long)group * rows, d);
                using var scores = new Tensor(Allocator, DType.Float32, kvHeads, (long)group * rows, len);
                if (!CudaHalfOperandGemm.TryAddmmBatch(scores, 0, scores, scale, q3, kt)) Ops.AddmmBatch(scores, 0, scores, scale, q3, kt);
                using Tensor perHead = scores.View(heads, rows, len);
                if (floor > 0)
                {
                    using Tensor below = perHead.Narrow(2, 0, Math.Min(floor, len));
                    Ops.Fill(below, float.NegativeInfinity);
                }
                if (mask.Pattern == MaskPattern.Causal) Ops.AddCausalMask(perHead, rows, len - nq + first, float.NegativeInfinity);
                Ops.Softmax(perHead, perHead);
                using var attn = new Tensor(Allocator, DType.Float32, kvHeads, (long)group * rows, d);
                if (!CudaHalfOperandGemm.TryAddmmBatch(attn, 0, attn, 1.0f, scores, vx)) Ops.AddmmBatch(attn, 0, attn, 1.0f, scores, vx);
                using Tensor attnHeads = attn.View(heads, rows, d);
                using Tensor permuted = attnHeads.Permute(1, 0, 2);
                using Tensor target = output.Narrow(0, first, rows);
                Ops.Copy(target, permuted);
            }
            return output;
        }

        /// <summary>Most bytes of attention scores a prompt chunk holds at once (its queries run in tiles).</summary>
        private const long ScoreBudgetBytes = 256L << 20;

        private CudaGraphKernels _kernels;
        private string _kernelsMissing;

        /// <summary>The executor's kernel module in this allocator's context, or null (with the reason
        /// reported once) where its PTX is missing or stale: those ops then run on the host.</summary>
        private CudaGraphKernels Kernels
        {
            get
            {
                if (_kernels != null || _kernelsMissing != null) return _kernels;
                var cuda = (CudaAllocator)Allocator;
                cuda.Context.MakeCurrent();
                if (!CudaGraphKernels.TryCreate(out _kernels, out _kernelsMissing))
                    Console.Error.WriteLine($"WARNING: CUDA graph kernels unavailable ({_kernelsMissing}); the gated delta rule runs on the host.");
                return _kernels;
            }
        }

        /// <summary>ggml's softplus as one kernel (the generic decomposition's abs/exp/log have no CUDA kernels).</summary>
        protected override Tensor Softplus(Tensor x)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || x.Storage is not CudaStorage || !x.IsContiguous() || x.ElementType != DType.Float32)
                return base.Softplus(x);
            var result = new Tensor(Allocator, DType.Float32, x.Sizes);
            Launch(new[] { x }, result, (k, stream) => k.LaunchSoftplus(Ptr(x), Ptr(result), x.ElementCount(), stream));
            return result;
        }

        /// <summary>ggml_ssm_conv as one kernel.</summary>
        protected override Tensor SsmConv(Tensor x, Tensor kernel)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || x.Storage is not CudaStorage || kernel.Storage is not CudaStorage
                || !x.IsContiguous() || !kernel.IsContiguous() || x.ElementType != DType.Float32 || kernel.ElementType != DType.Float32)
                return base.SsmConv(x, kernel);
            int ns = (int)x.Sizes[0], channels = (int)x.Sizes[1], length = (int)x.Sizes[2], width = (int)kernel.Sizes[1];
            var result = new Tensor(Allocator, DType.Float32, ns, length - width + 1, channels);
            Launch(new[] { x, kernel }, result, (k, stream) => k.LaunchSsmConv(Ptr(x), Ptr(kernel), Ptr(result), channels, length, width, ns, stream));
            return result;
        }

        protected override bool Fuses(GraphFusion kind) => kind is GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm && Kernels != null;

        /// <summary>The delta rule's prep and the gated norm after it as one kernel each.</summary>
        protected override bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            CudaGraphKernels kernels = Kernels;
            if (kernels == null || Array.Exists(inputs, t => t == null || t.Storage is not CudaStorage || !t.IsContiguous() || t.ElementType != DType.Float32))
                return false;
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            switch (kind)
            {
                case GraphFusion.DeltaNetPrep:
                {
                    int dk = (int)ip[0], kHeads = (int)ip[1], dv = (int)ip[2], vHeads = (int)ip[3];
                    Tensor window = inputs[0], qkv = inputs[1], convW = inputs[2];
                    int channels = (int)window.Sizes[0], convRows = (int)window.Sizes[^1], width = (int)convW.Sizes[^1];
                    int rows = (int)(qkv.ElementCount() / channels);
                    if (dk != dv || dk % 32 != 0 || dk > 1024 || channels != (2 * kHeads + vHeads) * dk
                        || window.ElementCount() != (long)channels * convRows || convW.ElementCount() != (long)channels * width
                        || inputs[3].ElementCount() != (long)rows * vHeads || inputs[4].ElementCount() != (long)rows * vHeads)
                        return false;
                    results[0] = new Tensor(Allocator, DType.Float32, channels, convRows + rows);
                    results[1] = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
                    results[2] = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
                    results[3] = new Tensor(Allocator, DType.Float32, rows, vHeads, dv);
                    results[4] = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
                    results[5] = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
                    Launch(inputs, results, (k, stream) => k.LaunchDeltaNetPrep(Ptr(inputs[0]), Ptr(inputs[1]), Ptr(inputs[2]), Ptr(inputs[3]),
                        Ptr(inputs[4]), Ptr(inputs[5]), Ptr(inputs[6]), Ptr(results[0]), Ptr(results[1]), Ptr(results[2]), Ptr(results[3]),
                        Ptr(results[4]), Ptr(results[5]), channels, convRows, width, dk, kHeads, vHeads, rows, eps, stream));
                    return true;
                }
                case GraphFusion.GatedRmsNorm:
                {
                    Tensor y = inputs[0];
                    int d = (int)inputs[2].ElementCount();
                    if (d % 32 != 0 || d > 1024 || y.ElementCount() % d != 0 || inputs[1].ElementCount() != y.ElementCount())
                        return false;
                    results[0] = new Tensor(Allocator, DType.Float32, y.Sizes);
                    int rows = (int)(y.ElementCount() / d);
                    Launch(inputs, results, (k, stream) => k.LaunchGatedRmsNorm(Ptr(inputs[0]), Ptr(inputs[1]), Ptr(inputs[2]), Ptr(results[0]),
                        rows, d, eps, stream));
                    return true;
                }
                default:
                    return false;
            }
        }

        private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

        /// <summary>Make the inputs and the output device-current, launch on the allocator's stream, mark the output written.</summary>
        private void Launch(Tensor[] inputs, Tensor output, Action<CudaGraphKernels, IntPtr> launch)
            => Launch(inputs, new[] { output }, launch);

        private void Launch(Tensor[] inputs, Tensor[] outputs, Action<CudaGraphKernels, IntPtr> launch)
        {
            var cuda = (CudaAllocator)Allocator;
            foreach (Tensor t in inputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            foreach (Tensor t in outputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            cuda.Context.MakeCurrent();
            launch(Kernels, cuda.Stream.Handle);
            foreach (Tensor t in outputs) ((CudaStorage)t.Storage).MarkDeviceModified();
        }

        /// <summary>ggml's gated delta rule on the device (one warp per state row).</summary>
        protected override Tensor GatedDeltaNet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
        {
            long[] nq = Ne(q), nv = Ne(v), ng = Ne(g);
            int s = (int)nv[0], heads = (int)nv[1], tokens = (int)nv[2], sequences = (int)nv[3], keyHeads = (int)nq[1];
            int slots = Math.Max(1, snapshots);
            bool kda = ng[0] == s && s > 1;
            CudaGraphKernels kernels = s % 32 == 0 && s <= 256 && nq[0] == s && heads % keyHeads == 0 ? Kernels : null;
            Tensor[] inputs = { q, k, v, g, beta, state };
            if (kernels == null || Array.Exists(inputs, t => t.Storage is not CudaStorage || !t.IsContiguous() || t.ElementType != DType.Float32))
                return base.GatedDeltaNet(q, k, v, g, beta, state, snapshots);

            long total = (long)sequences * tokens * heads * s + (long)slots * sequences * heads * s * s;
            var result = new Tensor(Allocator, DType.Float32, total);
            var cuda = (CudaAllocator)Allocator;
            foreach (Tensor t in inputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            var output = (CudaStorage)result.Storage;
            output.EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            cuda.Context.MakeCurrent();
            kernels.LaunchGatedDeltaNet(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(0),
                s, heads, keyHeads, tokens, sequences, slots, kda, cuda.Stream.Handle);
            output.MarkDeviceModified();
            return result;
        }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            var result = new Tensor(Allocator, DType.Float32, x.Sizes[0], weight.Ne1);
            if (CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, x, weight.CacheKey, weight.Data, weight.Format,
                    weight.Ne0, weight.Ne1, weight.Bytes))
                return result;
            result.Dispose();
            return HostProduct(weight, x);
        }
    }
}
