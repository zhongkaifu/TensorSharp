// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
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
            throw new NotSupportedException($"MLX has no product for weight type {weight.Format}");
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

            using Tensor qs = Ops.Mul(null, q, 1f / MathF.Sqrt(sv));
            using Tensor decay = Ops.Exp(null, g);
            MlxNative.MlxArray qa = default, ka = default, va = default, ga = default, ba = default, sa = default;
            MlxNative.MlxArray y = default, next = default, rows = default, order = default, picked = default;
            MlxNative.MlxArray yFlat = default, sFlat = default, all = default;
            try
            {
                qa = View(qs); ka = View(k); va = View(v); ga = View(decay); ba = View(beta); sa = View(state);
                if (slots == 1)
                {
                    MlxNative.GatedDelta(qa, ka, va, ga, ba, sa, ns, nt, hk, h, sv, sv, out y, out next);
                    sFlat = MlxNative.Reshape(next, new[] { ns * h * sv * sv });
                }
                else
                {
                    // Every row's state [T, B, Hv, Dv, Dk], newest first as ggml's snapshot slots.
                    MlxNative.GatedDeltaRows(qa, ka, va, ga, ba, sa, ns, nt, hk, h, sv, sv, out y, out next, out rows);
                    var reverse = new int[nt];
                    for (int t = 0; t < nt; t++) reverse[t] = nt - 1 - t;
                    unsafe
                    {
                        fixed (int* r = reverse) order = MlxNative.NewArrayFromHost((IntPtr)r, new[] { nt }, DType.Int32);
                    }
                    picked = MlxNative.TakeAxis(rows, order, 0);
                    sFlat = MlxNative.Reshape(picked, new[] { nt * ns * h * sv * sv });
                }
                yFlat = MlxNative.Reshape(y, new[] { ns * nt * h * sv });
                all = MlxNative.ConcatenateAxis(yFlat, sFlat, 0);
                long total = (long)ns * nt * h * sv + (long)slots * ns * h * sv * sv;
                var result = new Tensor(Allocator, DType.Float32, total);
                ((MlxStorage)result.Storage).ReplaceDeviceArray(all);
                all = default;
                return result;
            }
            finally
            {
                foreach (var a in new[] { qa, ka, va, ga, ba, sa, y, next, rows, order, picked, yFlat, sFlat, all })
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
