// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

using TensorSharp.Cpu;
using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// The managed CPU backend's executor plugin: programs run node by node on host tensors, quantized
    /// products through the managed GEMM kernels (the reference every other executor is checked against).
    /// </summary>
    public sealed class ManagedGraphExecutor : TensorGraphExecutor
    {
        [ModuleInitializer]
        internal static unsafe void Register()
        {
            GraphExecutors.Register(typeof(CpuAllocator), allocator => new ManagedGraphExecutor(allocator));
            // Backends without a kernel for a weight's format run it on the host (their per-op paths do).
            HostQuantizedProduct = (weight, x, rows, y) =>
            {
                fixed (float* xp = x)
                fixed (float* yp = y)
                    ManagedQuantizedOps.AddmmQuantizedToFloat32(weight.Format, weight.Data, weight.Ne0, weight.Ne1,
                        xp, (int)weight.Ne0, rows, yp, (int)weight.Ne1);
            };
        }

        public ManagedGraphExecutor(IAllocator allocator) : base(allocator) { }

        /// <summary>The base executor's capabilities, and a decode row's recurrence updates the slot's state in place (see
        /// <see cref="GatedDeltaNet"/>); decode steps replay their program a position on over windows padded to
        /// <see cref="DecodeWindowRows"/> rows, whose unwritten keys the attention stops before (see <see cref="FlashAttention"/>).</summary>
        public override GraphCapabilities Capabilities { get; } =
            new() { StridedViews = true, DecodeWindowRows = DecodeWindowRows, Replays = true, IndexedCacheWrites = true, InPlaceStateAlias = true, HostExperts = true };

        /// <summary>Decode window bucket: one program serves this many decode steps of a sequence.</summary>
        private const int DecodeWindowRows = 256;

        /// <summary>The recurrence wrote its result into the leaf already (<see cref="GatedDeltaNet"/>): nothing to copy.</summary>
        protected override bool TryAlias(Tensor result, Tensor leaf, long offsetBytes)
            => ReferenceEquals(result.Storage, leaf.Storage)
               && result.StorageOffset * result.ElementType.Size() == leaf.StorageOffset * leaf.ElementType.Size() + offsetBytes;

        /// <summary>Work (state elements touched in a step) below which the (sequence, head) pairs run serially.</summary>
        private const long ParallelRecurrenceWork = 1L << 16;

        /// <summary>
        /// ggml's gated delta rule (<see cref="TensorGraphExecutor.GatedDeltaNet"/>) on host memory, as Qwen35Model's managed
        /// recurrence runs it: each (sequence, head) state updated in place in its newest snapshot slot, SIMD dots and
        /// updates per state row, one exp per head and token (per key column with per-column decay), and the pairs in
        /// parallel once a step has enough work. The interpreter's reference copies every operand into arrays and runs
        /// scalar loops (3.3 ms a call on Qwen3.5-0.8B, against 0.2 ms here).
        /// </summary>
        protected override unsafe Tensor GatedDeltaNet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
        {
            int sv = (int)v.Sizes[3], h = (int)v.Sizes[2], nt = (int)v.Sizes[1], ns = (int)v.Sizes[0], hq = (int)q.Sizes[2];
            bool kda = g.Sizes[3] == sv && sv > 1;
            int slots = Math.Max(1, snapshots);
            long perHead = (long)sv * sv, attn = (long)sv * h * nt * ns, perSnap = perHead * h * ns, total = attn + perSnap * slots;
            // A decode row whose state is the tail of a [row outputs | state] buffer (a slot's backing): the result goes
            // there, the new state over the old (each (sequence, head) pair reads only its own state), and the alias that
            // follows finds it in place - a state copy per layer and token otherwise (1 MB a layer on Qwen3.5-0.8B).
            bool inPlace = slots == 1 && nt == 1 && ns == 1 && state.IsContiguous() && state.StorageOffset == attn
                && state.Storage.ElementCount == total && state.ElementType == DType.Float32;
            var result = inPlace
                ? new Tensor(new[] { total }, new[] { 1L }, state.Storage, 0)
                : new Tensor(Allocator, DType.Float32, total);
            float* outp = TensorComputePrimitives.GetFloatPointer(result);
            // A slot older than the step's tokens stays zero, as ggml leaves it.
            if (slots > nt)
                new Span<float>(outp + attn + perSnap * nt, checked((int)(perSnap * (slots - nt)))).Clear();
            long qa = (long)TensorComputePrimitives.GetFloatPointer(q), ka = (long)TensorComputePrimitives.GetFloatPointer(k);
            long va = (long)TensorComputePrimitives.GetFloatPointer(v), ga = (long)TensorComputePrimitives.GetFloatPointer(g);
            long ba = (long)TensorComputePrimitives.GetFloatPointer(beta), sa = (long)TensorComputePrimitives.GetFloatPointer(state);
            long oa = (long)outp;
            float scale = 1f / MathF.Sqrt(sv);

            void Pair(int pair)
            {
                int seq = pair / h, head = pair % h, qh = head % hq;
                float* s = (float*)oa + attn + ((long)seq * h + head) * perHead;
                float* from = (float*)sa + ((long)seq * h + head) * perHead;
                if (from != s)
                    Buffer.MemoryCopy(from, s, perHead * sizeof(float), perHead * sizeof(float));
                float* columns = stackalloc float[sv];
                float* decay = kda ? columns : null;
                var all = new Span<float>(s, (int)perHead);
                for (int t = 0; t < nt; t++)
                {
                    long qo = (((long)seq * nt + t) * hq + qh) * sv;
                    long vo = (((long)seq * nt + t) * h + head) * sv;
                    long go = (((long)seq * nt + t) * h + head) * (kda ? sv : 1);
                    float* kp = (float*)ka + qo, qp = (float*)qa + qo, vv = (float*)va + vo, o = (float*)oa + vo;
                    float b = ((float*)ba)[((long)seq * nt + t) * h + head];
                    if (kda) TensorPrimitives.Exp(new ReadOnlySpan<float>((float*)ga + go, sv), new Span<float>(decay, sv));
                    float e = kda ? 1f : MathF.Exp(((float*)ga)[go]);
                    for (int j = 0; j < sv; j++)
                        o[j] = DeltaRow(s + (long)j * sv, kp, qp, decay, e, vv[j], b, sv) * scale;
                    int slot = nt - 1 - t;
                    if (slot >= 1 && slot < slots)
                        all.CopyTo(new Span<float>((float*)oa + attn + slot * perSnap + ((long)seq * h + head) * perHead, (int)perHead));
                }
            }

            int pairs = ns * h;
            if (pairs > 1 && perHead * nt * pairs >= ParallelRecurrenceWork)
                System.Threading.Tasks.Parallel.For(0, pairs, Pair);
            else
                for (int pair = 0; pair < pairs; pair++) Pair(pair);
            return result;
        }

        protected override bool Fuses(GraphFusion kind) => kind is GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm;

        /// <summary>Tokens of a fused region below which it runs on the calling thread.</summary>
        private const int ParallelRegionRows = 8;

        /// <summary>
        /// The recurrent mixer's regions as single host passes (their semantics are MlxNative.GraphKernels'): a decode step's
        /// dozen small ops per Gated DeltaNet layer, each a few microseconds of dispatch on the host, become two loops.
        /// </summary>
        protected override unsafe bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            foreach (Tensor t in inputs)
                if (t == null || t.ElementType != DType.Float32) return false;
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            switch (kind)
            {
                case GraphFusion.DeltaNetPrep:
                    return DeltaNetPrep(inputs, (int)ip[0], (int)ip[1], (int)ip[2], (int)ip[3], eps, results);
                case GraphFusion.GatedRmsNorm:
                {
                    Tensor y = inputs[0], z = inputs[1], w = inputs[2];
                    int d = (int)w.ElementCount();
                    if (d == 0 || y.ElementCount() % d != 0 || z.ElementCount() != y.ElementCount()) return false;
                    int n = (int)(y.ElementCount() / d);
                    bool sigmoid = ip != null && ip.Length > 0 && ip[0] == 1;
                    var result = new Tensor(Allocator, DType.Float32, y.Sizes.ToArray());
                    long ya = (long)TensorComputePrimitives.GetFloatPointer(y), za = (long)TensorComputePrimitives.GetFloatPointer(z);
                    long wa = (long)TensorComputePrimitives.GetFloatPointer(w), ra = (long)TensorComputePrimitives.GetFloatPointer(result);
                    void Row(int r)
                    {
                        var yr = new ReadOnlySpan<float>((float*)ya + (long)r * d, d);
                        var zr = new ReadOnlySpan<float>((float*)za + (long)r * d, d);
                        var o = new Span<float>((float*)ra + (long)r * d, d);
                        float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(yr) / d + eps);
                        TensorPrimitives.Sigmoid(zr, o);
                        if (!sigmoid) TensorPrimitives.Multiply(zr, o, o);   // silu(z)
                        TensorPrimitives.Multiply(o, new ReadOnlySpan<float>((float*)wa, d), o);
                        TensorPrimitives.Multiply(o, yr, o);
                        TensorPrimitives.Multiply(o, inv, o);
                    }
                    if ((long)n * d >= ParallelRecurrenceWork) System.Threading.Tasks.Parallel.For(0, n, Row);
                    else for (int r = 0; r < n; r++) Row(r);
                    results[0] = result;
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <summary>GraphFusion.DeltaNetPrep: window [channels, convRows], qkv [rows, channels], conv weights [channels, width],
        /// alpha and beta [rows, vHeads], dt and a [vHeads] -> the conv input [channels, convRows + rows], q and k
        /// [rows, kHeads, dk] (silu of the causal conv, L2-normed per head: y / sqrt(sum y^2 + eps)), v [rows, vHeads, dv],
        /// g = softplus(alpha + dt) * a and beta = sigmoid(beta), [rows, vHeads, 1].</summary>
        private unsafe bool DeltaNetPrep(Tensor[] inputs, int dk, int kHeads, int dv, int vHeads, float eps, Tensor[] results)
        {
            Tensor window = inputs[0], qkv = inputs[1], convW = inputs[2];
            int channels = (int)window.Sizes[0], convRows = (int)window.Sizes[^1], width = (int)convW.Sizes[^1];
            int rows = (int)(qkv.ElementCount() / channels), span = convRows + rows, qk = kHeads * dk;
            if (channels != 2 * qk + vHeads * dv || width != convRows + 1 || window.ElementCount() != (long)channels * convRows
                || convW.ElementCount() != (long)channels * width || qkv.ElementCount() != (long)rows * channels
                || inputs[3].ElementCount() != (long)rows * vHeads || inputs[4].ElementCount() != (long)rows * vHeads
                || inputs[5].ElementCount() != vHeads || inputs[6].ElementCount() != vHeads)
                return false;
            var convIn = new Tensor(Allocator, DType.Float32, channels, span);
            var q = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
            var k = new Tensor(Allocator, DType.Float32, rows, kHeads, dk);
            var v = new Tensor(Allocator, DType.Float32, rows, vHeads, dv);
            var g = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
            var beta = new Tensor(Allocator, DType.Float32, rows, vHeads, 1);
            long wina = (long)TensorComputePrimitives.GetFloatPointer(window), qkva = (long)TensorComputePrimitives.GetFloatPointer(qkv);
            long cwa = (long)TensorComputePrimitives.GetFloatPointer(convW), cia = (long)TensorComputePrimitives.GetFloatPointer(convIn);
            long qa = (long)TensorComputePrimitives.GetFloatPointer(q), ka = (long)TensorComputePrimitives.GetFloatPointer(k);
            long va = (long)TensorComputePrimitives.GetFloatPointer(v), ga = (long)TensorComputePrimitives.GetFloatPointer(g);
            long ba = (long)TensorComputePrimitives.GetFloatPointer(beta);
            long ala = (long)TensorComputePrimitives.GetFloatPointer(inputs[3]), bra = (long)TensorComputePrimitives.GetFloatPointer(inputs[4]);
            long dta = (long)TensorComputePrimitives.GetFloatPointer(inputs[5]), aa = (long)TensorComputePrimitives.GetFloatPointer(inputs[6]);

            // Each token's conv output and its sigmoid in two rows of one scratch buffer.
            float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(checked(2 * rows * channels));
            try
            {
                fixed (float* sp = scratch)
                {
                    long ya = (long)sp;
                    // The conv input: each channel's window, then its rows of this step. One row (decode) also takes its
                    // conv in the same pass, the taps summed in order as the loop below sums them.
                    for (int c = 0; c < channels; c++)
                    {
                        float* dst = (float*)cia + (long)c * span, win = (float*)wina + (long)c * convRows;
                        if (rows == 1)
                        {
                            float* w = (float*)cwa + (long)c * width;
                            float x = 0f;
                            for (int p = 0; p < convRows; p++) { float wp = win[p]; dst[p] = wp; x += wp * w[p]; }
                            float last = ((float*)qkva)[c];
                            dst[convRows] = last;
                            ((float*)ya)[c] = x + last * w[convRows];
                            continue;
                        }
                        for (int p = 0; p < convRows; p++) dst[p] = win[p];
                        for (int t = 0; t < rows; t++) dst[convRows + t] = ((float*)qkva)[(long)t * channels + c];
                    }
                    void Token(int t)
                    {
                        float* y = (float*)ya + (long)t * channels, sig = (float*)ya + (long)(rows + t) * channels;
                        if (rows > 1)
                            for (int c = 0; c < channels; c++)
                            {
                                float* src = (float*)cia + (long)c * span + t, w = (float*)cwa + (long)c * width;
                                float x = 0f;
                                for (int j = 0; j < width; j++) x += src[j] * w[j];
                                y[c] = x;
                            }
                        var ys = new Span<float>(y, channels);
                        TensorPrimitives.Sigmoid(ys, new Span<float>(sig, channels));
                        TensorPrimitives.Multiply(ys, new ReadOnlySpan<float>(sig, channels), ys);   // silu
                        for (int head = 0; head < 2 * kHeads; head++)
                        {
                            var h = new ReadOnlySpan<float>(y + (long)head * dk, dk);
                            float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(h) + eps);
                            float* o = head < kHeads ? (float*)qa + ((long)t * kHeads + head) * dk : (float*)ka + ((long)t * kHeads + head - kHeads) * dk;
                            TensorPrimitives.Multiply(h, inv, new Span<float>(o, dk));
                        }
                        new ReadOnlySpan<float>(y + 2 * qk, vHeads * dv).CopyTo(new Span<float>((float*)va + (long)t * vHeads * dv, vHeads * dv));
                        for (int vh = 0; vh < vHeads; vh++)
                        {
                            float sv = ((float*)ala)[(long)t * vHeads + vh] + ((float*)dta)[vh];
                            float softplus = sv > 20f ? sv : MathF.Log(1f + MathF.Exp(sv));
                            ((float*)ga)[(long)t * vHeads + vh] = softplus * ((float*)aa)[vh];
                            ((float*)ba)[(long)t * vHeads + vh] = 1f / (1f + MathF.Exp(-((float*)bra)[(long)t * vHeads + vh]));
                        }
                    }
                    if (rows >= ParallelRegionRows) System.Threading.Tasks.Parallel.For(0, rows, Token);
                    else for (int t = 0; t < rows; t++) Token(t);
                }
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(scratch);
            }
            results[0] = convIn; results[1] = q; results[2] = k; results[3] = v; results[4] = g; results[5] = beta;
            return true;
        }

        /// <summary>
        /// One query row over F32 or F16 keys it sees all of (no mask, or causal to the last key): the grouped decode kernel the
        /// per-op layers use (<see cref="ModelBase.AttentionDecodeGrouped"/>: each KV head's keys read once for its query
        /// group, SIMD dots, the sequence split over cores with an online softmax), reading the cache window in place. Other
        /// shapes take the interpreter's attention.
        /// </summary>
        protected override unsafe Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            long heads = q.Sizes[0], d = q.Sizes[2], kvHeads = k.Sizes[0], keys = k.Sizes[1];
            // A padded decode window (its unwritten tail masked) runs its pattern over the written keys only.
            bool trimmed = mask.Pattern == MaskPattern.Explicit && mask.Trimmed != MaskPattern.Explicit && mask.Keys > 0;
            MaskPattern pattern = trimmed ? mask.Trimmed : mask.Pattern;
            int floor = trimmed ? mask.TrimmedFloor : mask.Floor;
            bool open = floor == 0 && (pattern == MaskPattern.None || (pattern == MaskPattern.Causal && q.Sizes[1] == 1));
            bool half = k.ElementType == DType.Float16 && v.ElementType == DType.Float16;
            if (q.Sizes[1] != 1 || !open || v.Sizes[2] != d || k.Sizes[2] != d || kvHeads == 0 || heads % kvHeads != 0
                || q.ElementType != DType.Float32 || !(half || (k.ElementType == DType.Float32 && v.ElementType == DType.Float32))
                || k.Strides[2] != 1 || k.Strides[1] != d || v.Strides[2] != 1 || v.Strides[1] != d
                || k.Strides[0] != v.Strides[0] || k.Strides[0] % d != 0 || v.Sizes[1] != keys)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            if (trimmed) keys = mask.Keys;
            Tensor packed = q.IsContiguous() ? null : Ops.NewContiguous(q);
            try
            {
                var result = new Tensor(Allocator, DType.Float32, 1, heads, d);
                int stride = checked((int)(k.Strides[0] / d));
                float* qp = TensorComputePrimitives.GetFloatPointer(packed ?? q), rp = TensorComputePrimitives.GetFloatPointer(result);
                if (half)
                    ModelBase.AttentionDecodeGroupedF16(qp, TensorComputePrimitives.GetHalfPointer(k), TensorComputePrimitives.GetHalfPointer(v),
                        rp, (int)heads, (int)kvHeads, (int)d, stride, checked((int)keys), scale);
                else
                    ModelBase.AttentionDecodeGrouped(qp, TensorComputePrimitives.GetFloatPointer(k), TensorComputePrimitives.GetFloatPointer(v),
                        rp, (int)heads, (int)kvHeads, (int)d, stride, checked((int)keys), scale, Environment.ProcessorCount);
                return result;
            }
            finally
            {
                packed?.Dispose();
            }
        }

        /// <summary>One state row of the delta rule in two passes over it: the decay and its dot with k, then the update by k
        /// times the row's delta and its dot with q (returned). The row stays in L1 between them.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DeltaRow(float* row, float* k, float* q, float* decay, float e, float v, float b, int n)
        {
            int w = System.Numerics.Vector<float>.Count, i = 0;
            var acc = System.Numerics.Vector<float>.Zero;
            var ev = new System.Numerics.Vector<float>(e);
            for (; i <= n - w; i += w)
            {
                var r = TensorComputePrimitives.LoadVector(row + i) * (decay != null ? TensorComputePrimitives.LoadVector(decay + i) : ev);
                TensorComputePrimitives.StoreVector(row + i, r);
                acc += r * TensorComputePrimitives.LoadVector(k + i);
            }
            float dk = System.Numerics.Vector.Sum(acc);
            for (; i < n; i++) { row[i] *= decay != null ? decay[i] : e; dk += row[i] * k[i]; }
            float delta = (v - dk) * b;
            var dv = new System.Numerics.Vector<float>(delta);
            acc = System.Numerics.Vector<float>.Zero;
            for (i = 0; i <= n - w; i += w)
            {
                var r = TensorComputePrimitives.LoadVector(row + i) + TensorComputePrimitives.LoadVector(k + i) * dv;
                TensorComputePrimitives.StoreVector(row + i, r);
                acc += r * TensorComputePrimitives.LoadVector(q + i);
            }
            float dq = System.Numerics.Vector.Sum(acc);
            for (; i < n; i++) { row[i] += k[i] * delta; dq += row[i] * q[i]; }
            return dq;
        }

        /// <summary>ggml_ssm_conv (<see cref="TensorGraphExecutor.SsmConv"/>) as one pass over host memory: each output is
        /// its taps summed in order, as the reference's per-tap multiply and add sum them.</summary>
        protected override unsafe Tensor SsmConv(Tensor x, Tensor kernel)
        {
            long ns = x.Sizes[0], channels = x.Sizes[1], length = x.Sizes[2], width = kernel.Sizes[1];
            long nt = length - width + 1;
            var result = new Tensor(Allocator, DType.Float32, ns, nt, channels);
            long xa = (long)TensorComputePrimitives.GetFloatPointer(x), wa = (long)TensorComputePrimitives.GetFloatPointer(kernel);
            long ra = (long)TensorComputePrimitives.GetFloatPointer(result);

            void Channel(long index)
            {
                long seq = index / channels, c = index % channels;
                float* xs = (float*)xa + (seq * channels + c) * length;
                float* w = (float*)wa + c * width;
                float* r = (float*)ra + seq * nt * channels + c;
                for (long t = 0; t < nt; t++)
                {
                    float sum = xs[t] * w[0];
                    for (long kk = 1; kk < width; kk++) sum += xs[t + kk] * w[kk];
                    r[t * channels] = sum;
                }
            }

            long rows = ns * channels;
            if (rows * nt * width >= ParallelRecurrenceWork * 4)
                System.Threading.Tasks.Parallel.For(0L, rows, Channel);
            else
                for (long i = 0; i < rows; i++) Channel(i);
            return result;
        }

        protected override unsafe Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            if (weight.Data == IntPtr.Zero)
                throw new InvalidOperationException("a quantized weight's host bytes were released");
            int rows = (int)x.Sizes[0];
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            ManagedQuantizedOps.AddmmQuantizedToFloat32(weight.Format, weight.Data, weight.Ne0, weight.Ne1,
                TensorComputePrimitives.GetFloatPointer(x), (int)weight.Ne0, rows,
                TensorComputePrimitives.GetFloatPointer(result), (int)weight.Ne1);
            return result;
        }
    }
}
