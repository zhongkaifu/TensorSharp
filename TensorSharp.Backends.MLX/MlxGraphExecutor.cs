// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using TensorSharp.Graph;

namespace TensorSharp.MLX
{
    /// <summary>
    /// MLX's executor plugin: programs run node by node on MLX arrays (lazily, as MLX evaluates), with
    /// MLX's quantized products and fused attention.
    /// </summary>
    public sealed class MlxGraphExecutor : TensorGraphExecutor
    {
        [ModuleInitializer]
        internal static void Register()
            => GraphExecutors.Register(typeof(MlxAllocator), allocator => new MlxGraphExecutor(allocator));

        public MlxGraphExecutor(IAllocator allocator) : base(allocator) { }

        /// <summary>The residual add and the next RMSNorm as MLX's fused kernel (the per-op layer's
        /// TryAddRmsNorm).</summary>
        protected override bool TryAddWeightedNorm(Tensor a, Tensor b, Tensor weight, float eps, out Tensor sum, out Tensor normed)
        {
            sum = normed = null;
            long cols = a.Sizes[^1], rows = a.ElementCount() / cols;
            if (!(a.Storage is MlxStorage && b.Storage is MlxStorage && weight.Storage is MlxStorage)
                || a.ElementType != DType.Float32 || b.ElementType != DType.Float32 || weight.ElementType != DType.Float32
                || weight.ElementCount() != cols || !a.IsContiguous() || !b.IsContiguous() || !weight.IsContiguous())
                return false;
            MlxNative.MlxArray av = default, bv = default, wv = default, s = default, n = default;
            try
            {
                av = ((MlxStorage)a.Storage).CreateArrayView(new[] { (int)rows, (int)cols }, new[] { cols, 1L }, a.StorageOffset);
                bv = ((MlxStorage)b.Storage).CreateArrayView(new[] { (int)rows, (int)cols }, new[] { cols, 1L }, b.StorageOffset);
                wv = ((MlxStorage)weight.Storage).CreateArrayView(new[] { (int)cols }, new[] { 1L }, weight.StorageOffset);
                (s, n) = MlxNative.AddRmsNorm(av, bv, wv, eps, (int)rows, (int)cols);
                sum = new Tensor(Allocator, DType.Float32, a.Sizes);
                ((MlxStorage)sum.Storage).ReplaceDeviceArray(MlxNative.Reshape(s, ToInts(a.Sizes)));
                normed = new Tensor(Allocator, DType.Float32, a.Sizes);
                ((MlxStorage)normed.Storage).ReplaceDeviceArray(MlxNative.Reshape(n, ToInts(a.Sizes)));
                return true;
            }
            finally
            {
                foreach (var x in new[] { av, bv, wv, s, n }) MlxNative.FreeArray(x);
            }
        }

        private static int[] ToInts(ReadOnlySpan<long> sizes)
        {
            var r = new int[sizes.Length];
            for (int i = 0; i < r.Length; i++) r[i] = checked((int)sizes[i]);
            return r;
        }

        protected override bool Fuses(GraphFusion kind) => kind is GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm or GraphFusion.SwigluFfn;

        /// <summary>The delta rule's prep and the gated norm after it as one Metal kernel each
        /// (MlxNative.GraphKernels).</summary>
        protected override bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            if (kind == GraphFusion.SwigluFfn) return TrySwigluFfn(inputs[0], weights, ip, results);
            foreach (Tensor t in inputs)
                if (t == null || !(t.Storage is MlxStorage) || t.ElementType != DType.Float32) return false;
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            var views = new MlxNative.MlxArray[inputs.Length];
            MlxNative.MlxArray[] outs = null;
            try
            {
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
                        // The kernel reads each input in its documented 2-D (or 1-D) shape, through the view's own
                        // strides (a prompt chunk's row blocks of the packed projection are not copied).
                        int[][] shapes =
                        {
                            new[] { channels, convRows }, new[] { rows, channels }, new[] { channels, width },
                            new[] { rows, vHeads }, new[] { rows, vHeads }, new[] { vHeads }, new[] { vHeads },
                        };
                        for (int i = 0; i < inputs.Length; i++) views[i] = StridedView(inputs[i], shapes[i]);
                        outs = MlxNative.DeltaNetPrep(views[0], views[1], views[2], views[3], views[4], views[5], views[6],
                            rows, channels, convRows, width, dk, kHeads, vHeads, eps);
                        long[][] sizes =
                        {
                            new long[] { channels, convRows + rows }, new long[] { rows, kHeads, dk }, new long[] { rows, kHeads, dk },
                            new long[] { rows, vHeads, dv }, new long[] { rows, vHeads, 1 }, new long[] { rows, vHeads, 1 },
                        };
                        for (int o = 0; o < results.Length; o++) results[o] = Wrap(outs[o], sizes[o]);
                        return true;
                    }
                    case GraphFusion.GatedRmsNorm:
                    {
                        for (int i = 0; i < inputs.Length; i++) views[i] = View(inputs[i]);
                        Tensor y = inputs[0], w = inputs[2];
                        int d = (int)w.ElementCount();
                        if (d % 32 != 0 || d > 1024 || y.ElementCount() % d != 0 || inputs[1].ElementCount() != y.ElementCount())
                            return false;
                        int n = (int)(y.ElementCount() / d);
                        outs = new[] { MlxNative.GatedRmsNorm(views[0], views[1], views[2], n, d, eps) };
                        results[0] = Wrap(outs[0], y.Sizes);
                        return true;
                    }
                    default:
                        return false;
                }
            }
            finally
            {
                foreach (var v in views) MlxNative.FreeArray(v);
                if (outs != null) foreach (var o in outs) MlxNative.FreeArray(o);
            }
        }

        /// <summary>A prompt chunk's SwiGLU feed-forward in half precision (MlxQuantizedOps.TrySwiGluFfnHalf): the
        /// activations stay F16 between the three products. Fewer rows (decode, a verify) run the ops.</summary>
        private bool TrySwigluFfn(Tensor x, IGraphWeight[] weights, long[] ip, Tensor[] results)
        {
            if (x == null || x.Storage is not MlxStorage || x.ElementType != DType.Float32 || x.ElementCount() % x.Sizes[^1] != 0)
                return false;
            bool split = ip != null && ip.Length > 0 && ip[0] == 1;
            IGraphWeight gate = weights[1], up = split ? weights[2] : null, down = weights[split ? 3 : 2];
            if (gate == null || down == null || (split && up == null)) return false;
            long rows = x.ElementCount() / x.Sizes[^1];
            var result = new Tensor(Allocator, DType.Float32, rows, down.Ne1);
            if (MlxQuantizedOps.TrySwiGluFfnHalf(result, x, Spec(gate), up == null ? default : Spec(up), Spec(down)))
            {
                results[0] = result;
                return true;
            }
            result.Dispose();
            return false;
        }

        private static MlxQuantizedOps.WeightSpec Spec(IGraphWeight w) => new(w.CacheKey, w.Data, w.Format, w.Ne0, w.Ne1, w.Bytes);

        /// <summary>A view of <paramref name="t"/> in <paramref name="shape"/> (its own extents with leading ones
        /// dropped) over its strides, or a reshape where its layout is not of that shape.</summary>
        private static MlxNative.MlxArray StridedView(Tensor t, int[] shape)
        {
            var sizes = new List<long>();
            var strides = new List<long>();
            for (int d = 0; d < t.DimensionCount; d++)
            {
                if (t.Sizes[d] == 1) continue;
                sizes.Add(t.Sizes[d]);
                strides.Add(t.Strides[d]);
            }
            bool same = sizes.Count <= shape.Length;
            for (int d = 0, k = shape.Length - sizes.Count; same && d < sizes.Count; d++) same = sizes[d] == shape[k + d];
            if (same)
            {
                var full = new long[shape.Length];
                for (int d = shape.Length - 1, k = sizes.Count - 1; d >= 0; d--, k--)
                    full[d] = k >= 0 ? strides[k] : (d + 1 < shape.Length ? full[d + 1] * shape[d + 1] : 1);
                return ((MlxStorage)t.Storage).CreateArrayView(shape, full, t.StorageOffset);
            }
            MlxNative.MlxArray view = View(t);
            try { return MlxNative.Reshape(view, shape); }
            finally { MlxNative.FreeArray(view); }
        }

        /// <summary>An aliased result (a verify window's [outputs | states] into the request's buffer) is adopted
        /// by the leaf's storage as views of the result's arrays: no copy.</summary>
        protected override bool TryAlias(Tensor result, Tensor leaf, long offsetBytes)
        {
            if (!(result.Storage is MlxStorage source && leaf.Storage is MlxStorage target) || ReferenceEquals(source, target)
                || result.ElementType != DType.Float32 || leaf.ElementType != DType.Float32 || !result.IsContiguous()
                || offsetBytes % sizeof(float) != 0)
                return false;
            long offset = leaf.StorageOffset + offsetBytes / sizeof(float), n = result.ElementCount();
            if (offset < 0 || offset + n > target.ElementCount) return false;
            target.AdoptRange(offset, source.FlatRange(result.StorageOffset, n));
            return true;
        }

        /// <summary>A copy over a whole contiguous tensor (a recurrent state's update) takes the source's
        /// array: MLX arrays are immutable, so the destination reads the same bytes with no copy.</summary>
        protected override void CopyInto(Tensor dst, Tensor src)
        {
            if (dst.Storage is MlxStorage target && src.Storage is MlxStorage source && !ReferenceEquals(target, source)
                && dst.ElementType == src.ElementType && dst.StorageOffset == 0 && dst.ElementCount() == target.ElementCount
                && dst.IsContiguous() && src.IsContiguous() && src.ElementCount() == dst.ElementCount())
            {
                target.ReplaceDeviceArray(source.CreateArrayView(new[] { checked((int)src.ElementCount()) }, new[] { 1L }, src.StorageOffset));
                return;
            }
            base.CopyInto(dst, src);
        }

        /// <summary>MLX ops run on its worker thread: the whole plan runs there, so its ops do not hop.</summary>
        protected override bool OnBackendThread(Func<bool> run) => MlxWorker.Shared.Invoke(run);

        /// <summary>MLX ops take strided array views.</summary>
        protected override bool KernelsReadStridedViews => true;

        /// <summary>Layers per asynchronous evaluation: the GPU runs them while later layers are built
        /// (the per-op runtime's measured cadence: Qwen3.5-0.8B 191 tok/s at 4 vs 183 at 8).</summary>
        private const int EvalEveryLayers = 4;

        protected override void AtBoundary(Tensor value, int index, int count)
        {
            if (index % EvalEveryLayers == 0 && index != count)
                MlxFusedOps.TryAsyncEvaluate(value);
        }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            var result = new Tensor(Allocator, DType.Float32, x.Sizes[0], weight.Ne1);
            // MLX's quantized kernels, else (Q2_K, Q3_K, BF16) the raw weight dequantized to exact F32.
            if (MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, x, weight.CacheKey, weight.Data, weight.Format,
                    weight.Ne0, weight.Ne1, weight.Bytes)
                || (MlxQuantizedOps.CanPreloadDirectFloatType(weight.Format)
                    && MlxQuantizedOps.TryAddmmDirectFloat32(result, x, weight.CacheKey, weight.Data, weight.Format,
                        weight.Ne0, weight.Ne1, weight.Bytes)))
                return result;
            result.Dispose();
            return HostProduct(weight, x);
        }

        /// <summary>MLX's fused attention (GQA native) over the K/V views as they are: q is cast to the
        /// cache's type, a causal or empty mask runs as MLX's own mode (keys before the floor left out
        /// of the view), any other mask as an array. The output [1, heads, nq, dv] is transposed to
        /// ggml's [nq, heads, dv].</summary>
        protected override Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            if (!(q.Storage is MlxStorage && k.Storage is MlxStorage && v.Storage is MlxStorage)
                || k.ElementType != v.ElementType || q.ElementType != DType.Float32
                || (k.ElementType != DType.Float32 && k.ElementType != DType.Float16))
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            long heads = q.Sizes[0], nq = q.Sizes[1], dv = v.Sizes[2];
            int floor = mask.Floor;
            using Tensor kv = floor > 0 ? k.Narrow(1, floor, k.Sizes[1] - floor) : null;
            using Tensor vv = floor > 0 ? v.Narrow(1, floor, v.Sizes[1] - floor) : null;
            Tensor keys = kv ?? k, values = vv ?? v;
            DType cache = k.ElementType;
            MlxNative.MlxArray qa = default, qt = default, ka = default, va = default, ma = default, mt = default;
            MlxNative.MlxArray att = default, rows = default, typed = default, flat = default;
            try
            {
                qa = Head4(q);
                ka = Head4(keys);
                va = Head4(values);
                MlxNative.MlxArray query = qa;
                if (cache != DType.Float32) query = qt = MlxNative.Astype(qa, cache);
                string mode = mask.Pattern switch
                {
                    MaskPattern.None => string.Empty,
                    MaskPattern.Causal => nq == 1 ? string.Empty : "causal",
                    _ => "array",
                };
                if (mask.Pattern == MaskPattern.Explicit)
                {
                    ma = ((MlxStorage)mask.Tensor.Storage).CreateArrayView(new[] { 1, 1, (int)nq, (int)mask.Tensor.Sizes[1] },
                        new[] { 0L, 0L, mask.Tensor.Strides[0], mask.Tensor.Strides[1] }, mask.Tensor.StorageOffset);
                    if (cache != DType.Float32) mt = MlxNative.Astype(ma, cache);
                }
                att = MlxNative.FastScaledDotProductAttention(query, ka, va, scale, mode, mt.IsValid ? mt : ma);
                rows = MlxNative.Transpose(att, new[] { 0, 2, 1, 3 });
                typed = cache != DType.Float32 ? MlxNative.Astype(rows, DType.Float32) : MlxNative.Contiguous(rows);
                flat = MlxNative.Reshape(typed, new[] { (int)nq, (int)heads, (int)dv });
                var result = new Tensor(Allocator, DType.Float32, nq, heads, dv);
                ((MlxStorage)result.Storage).ReplaceDeviceArray(flat);
                flat = default;
                return result;
            }
            finally
            {
                foreach (var a in new[] { qa, qt, ka, va, ma, mt, att, rows, typed, flat }) MlxNative.FreeArray(a);
            }
        }

        /// <summary>[h, n, d] (any strides) as an MLX view [1, h, n, d].</summary>
        private static MlxNative.MlxArray Head4(Tensor t)
            => ((MlxStorage)t.Storage).CreateArrayView(new[] { 1, (int)t.Sizes[0], (int)t.Sizes[1], (int)t.Sizes[2] },
                new[] { t.Strides[0] * t.Sizes[0], t.Strides[0], t.Strides[1], t.Strides[2] }, t.StorageOffset);

        /// <summary>
        /// ggml's gated delta rule on MLX's Metal kernels (the T=1 decode kernel, the blocked prompt
        /// kernel, or the per-row kernel that keeps every row's state for a verify window). The kernels
        /// take the decay factor exp(g) and no output scale, so q carries ggml's 1/sqrt(S).
        /// </summary>
        protected override Tensor GatedDeltaNet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
        {
            long[] nq = Ne(q), nv = Ne(v), ng = Ne(g);
            int sv = (int)nv[0], h = (int)nv[1], nt = (int)nv[2], ns = (int)nv[3], hk = (int)nq[1];
            int slots = Math.Max(1, snapshots);
            if (ng[0] != 1 || nq[0] != sv || sv % 32 != 0 || h % hk != 0 || (slots != 1 && slots != nt)
                || !(q.Storage is MlxStorage))
                return base.GatedDeltaNet(q, k, v, g, beta, state, snapshots);

            // One decode row takes ggml's q and log decay as they are; other shapes the kernels' (q / sqrt(S), exp(g)).
            bool decode = slots == 1 && nt == 1;
            using Tensor qs = decode ? null : Ops.Mul(null, q, 1f / MathF.Sqrt(sv));
            using Tensor decay = decode ? null : Ops.Exp(null, g);
            MlxNative.MlxArray qa = default, ka = default, va = default, ga = default, ba = default, sa = default;
            MlxNative.MlxArray y = default, next = default, rows = default, order = default, picked = default;
            MlxNative.MlxArray yFlat = default, sFlat = default;
            try
            {
                qa = View(qs ?? q); ka = View(k); va = View(v); ga = View(decay ?? g); ba = View(beta); sa = View(state);
                if (decode)
                {
                    (y, next) = MlxNative.GatedDeltaDecode(qa, ka, va, ga, ba, sa, ns, hk, h, sv, sv);
                    sFlat = MlxNative.Reshape(next, new[] { ns * h * sv * sv });
                }
                else if (slots == 1)
                {
                    MlxNative.GatedDelta(qa, ka, va, ga, ba, sa, ns, nt, hk, h, sv, sv, out y, out next);
                    sFlat = MlxNative.Reshape(next, new[] { ns * h * sv * sv });
                }
                else
                {
                    // Every row's state [T, B, Hv, Dv, Dk]: ggml's snapshot slots are newest first, so the
                    // result holds each row's state as its own part in that order (slices, no gather).
                    MlxNative.GatedDeltaRows(qa, ka, va, ga, ba, sa, ns, nt, hk, h, sv, sv, out y, out next, out rows);
                    yFlat = MlxNative.Reshape(y, new[] { ns * nt * h * sv });
                    int per = ns * h * sv * sv;
                    picked = MlxNative.Reshape(rows, new[] { nt * per });
                    var parts = new MlxNative.MlxArray[1 + nt];
                    parts[0] = yFlat;
                    for (int t = 0; t < nt; t++)
                        parts[1 + t] = MlxNative.Slice(picked, new[] { (nt - 1 - t) * per }, new[] { (nt - t) * per }, new[] { 1 });
                    var rowsResult = new Tensor(Allocator, DType.Float32, (long)ns * nt * h * sv + (long)nt * per);
                    ((MlxStorage)rowsResult.Storage).ReplaceDeviceParts(parts);
                    yFlat = default;
                    return rowsResult;
                }
                yFlat = MlxNative.Reshape(y, new[] { ns * nt * h * sv });
                long total = (long)ns * nt * h * sv + (long)slots * ns * h * sv * sv;
                var result = new Tensor(Allocator, DType.Float32, total);
                // ggml's [outputs | states] as two parts: the views of either read it without a concat.
                ((MlxStorage)result.Storage).ReplaceDeviceParts(yFlat, sFlat);
                yFlat = sFlat = default;
                return result;
            }
            finally
            {
                foreach (var a in new[] { qa, ka, va, ga, ba, sa, y, next, rows, order, picked, yFlat, sFlat })
                    MlxNative.FreeArray(a);
            }
        }

        /// <summary>softplus(x) = logaddexp(x, 0): one MLX op.</summary>
        protected override Tensor Softplus(Tensor x)
        {
            if (!(x.Storage is MlxStorage) || x.ElementType != DType.Float32) return base.Softplus(x);
            MlxNative.MlxArray xa = default, zero = default, r = default;
            try
            {
                xa = View(x);
                zero = MlxNative.NewScalar(0f);
                r = MlxNative.LogAddExp(xa, zero);
                return Wrap(r, x.Sizes);
            }
            finally
            {
                foreach (var a in new[] { xa, zero, r }) MlxNative.FreeArray(a);
            }
        }

        /// <summary>ggml_ssm_conv as MLX's depthwise conv1d: x [n_s, channels, width - 1 + n_t] read
        /// channels-last, the kernel [channels, width] as [channels, width, 1], groups = channels; the
        /// output [n_s, n_t, channels] is ggml's layout.</summary>
        protected override Tensor SsmConv(Tensor x, Tensor kernel)
        {
            if (!(x.Storage is MlxStorage && kernel.Storage is MlxStorage) || x.ElementType != DType.Float32 || kernel.ElementType != DType.Float32)
                return base.SsmConv(x, kernel);
            long ns = x.Sizes[0], channels = x.Sizes[1], length = x.Sizes[2], width = kernel.Sizes[1];
            MlxNative.MlxArray input = default, weight = default, r = default;
            try
            {
                input = ((MlxStorage)x.Storage).CreateArrayView(new[] { (int)ns, (int)length, (int)channels },
                    new[] { x.Strides[0], x.Strides[2], x.Strides[1] }, x.StorageOffset);
                weight = ((MlxStorage)kernel.Storage).CreateArrayView(new[] { (int)channels, (int)width, 1 },
                    new[] { kernel.Strides[0], kernel.Strides[1], 1L }, kernel.StorageOffset);
                r = MlxNative.Conv1d(input, weight, 1, 0, 1, (int)channels);
                return Wrap(r, new[] { ns, length - width + 1, channels });
            }
            finally
            {
                foreach (var a in new[] { input, weight, r }) MlxNative.FreeArray(a);
            }
        }

        /// <summary>A tensor of <paramref name="sizes"/> over <paramref name="array"/> (the caller still frees
        /// its handle).</summary>
        private Tensor Wrap(MlxNative.MlxArray array, ReadOnlySpan<long> sizes)
        {
            var t = new Tensor(Allocator, DType.Float32, sizes);
            ((MlxStorage)t.Storage).ReplaceDeviceArray(MlxNative.Reshape(array, ToInts(sizes)));
            return t;
        }

        private static MlxNative.MlxArray View(Tensor t) => ((MlxStorage)t.Storage).CreateArrayView(t);

    }
}
