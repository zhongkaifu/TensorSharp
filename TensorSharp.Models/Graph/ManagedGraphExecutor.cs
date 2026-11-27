// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

using TensorSharp.Cpu;
using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// The managed CPU backend's executor plugin: programs run node by node on host tensors, quantized
    /// products through the managed GEMM kernels (the reference every other executor is checked against).
    /// </summary>
    public sealed partial class ManagedGraphExecutor : TensorGraphExecutor
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
            HostQuantizedRows = (weight, rows, y) =>
            {
                long rowBytes = weight.Bytes / (weight.Ne1 * Math.Max(1, weight.Ne2));
                fixed (float* yp = y)
                    for (int i = 0; i < rows.Length; i++)
                        ManagedQuantizedOps.DequantizeToFloat32Native(weight.Format, weight.Data + (nint)(rows[i] * rowBytes),
                            (IntPtr)(yp + (long)i * weight.Ne0), weight.Ne0);
            };
        }

        public ManagedGraphExecutor(IAllocator allocator) : base(allocator) { }

        // Whether this run (one step) has set the worker pool's width yet: its first weight product does (decode-sized
        // steps run narrower, see CpuWorkerPool.SelectWidthForStep). A step without weight products keeps the width it finds.
        private bool _stepWidthSet;

        protected override void BeginRun()
        {
            base.BeginRun();
            _stepWidthSet = false;
        }

        /// <summary>The step's first weight product picks the worker pool's width for the rest of the step (and after it,
        /// until a step picks again): a later product of other rows in the same step - an expert's few routed rows, the pruned
        /// last layer's output rows - does not switch it.</summary>
        private void SetStepWidth(long rows)
        {
            if (_stepWidthSet) return;
            _stepWidthSet = true;
            // A program of whole-image products (a diffusion denoiser's step) is never decode-sized, though its first product
            // may be (the timestep embedding's rows).
            CpuWorkerPool.SelectWidthForStep(WideTokenProducts ? long.MaxValue : rows);
        }

        protected override void DenseProduct(Tensor result, Tensor rows, Tensor weight)
        {
            SetStepWidth(rows.Sizes[0]);
            base.DenseProduct(result, rows, weight);
        }

        /// <summary>The base executor's capabilities, and a decode row's recurrence updates the slot's state in place (see
        /// <see cref="GatedDeltaNet"/>); decode steps replay their program a position on over windows padded to
        /// <see cref="DecodeWindowRows"/> rows, whose unwritten keys the attention stops before (see <see cref="FlashAttention"/>).</summary>
        public override GraphCapabilities Capabilities => StructuralMasksForTests ? s_structuralMaskCapabilities : s_capabilities;

        private static readonly GraphCapabilities s_capabilities =
            new() { StridedViews = true, DecodeWindowRows = DecodeWindowRows, Replays = true, IndexedCacheWrites = true, InPlaceStateAlias = true, HostExperts = true };

        private static readonly GraphCapabilities s_structuralMaskCapabilities = s_capabilities with { StructuralCausalMasks = true };

        /// <summary>Tests: prompt chunks' causal masks arrive as their pattern (<see cref="GraphCapabilities.StructuralCausalMasks"/>),
        /// as on the CUDA executor; this executor's attention reads their values, which the interpreter then makes.</summary>
        internal bool StructuralMasksForTests { get; set; }

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

        /// <summary><see cref="GraphFusion.PairRotary"/> in one pass over the rows: the region's products and sums, each
        /// rounded as its ops round them.</summary>
        private unsafe bool PairRotary(Tensor x, Tensor cos, Tensor sin, Tensor[] results)
        {
            if (x.DimensionCount != 3 || !x.IsContiguous() || !cos.IsContiguous() || !sin.IsContiguous()) return false;
            long rows = x.Sizes[0], heads = x.Sizes[1], d = x.Sizes[2], half = d / 2;
            if (d % 2 != 0 || cos.ElementCount() != rows * half || sin.ElementCount() != rows * half) return false;
            var result = new Tensor(Allocator, DType.Float32, rows, heads, d);
            long xa = (long)TensorComputePrimitives.GetFloatPointer(x), ra = (long)TensorComputePrimitives.GetFloatPointer(result);
            long ca = (long)TensorComputePrimitives.GetFloatPointer(cos), sa = (long)TensorComputePrimitives.GetFloatPointer(sin);
            void Row(long r)
            {
                float* c = (float*)ca + r * half, s = (float*)sa + r * half;
                for (long h = 0; h < heads; h++)
                {
                    float* xi = (float*)xa + (r * heads + h) * d, o = (float*)ra + (r * heads + h) * d;
                    for (long j = 0; j < half; j++)
                    {
                        float even = xi[2 * j], odd = xi[2 * j + 1];
                        float ec = even * c[j], os = odd * s[j], oc = odd * c[j], es = even * s[j];
                        o[j] = ec - os;
                        o[half + j] = oc + es;
                    }
                }
            }
            if (rows >= ParallelRegionRows) System.Threading.Tasks.Parallel.For(0L, rows, Row);
            else for (long r = 0; r < rows; r++) Row(r);
            results[0] = result;
            return true;
        }

        protected override bool Fuses(GraphFusion kind) => kind is GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm or GraphFusion.PairRotary
            or GraphFusion.FullAttention
            || (kind == GraphFusion.LoraLinear && ServesLoraLinearInPlace) || (kind == GraphFusion.DitAttention && ServesDitAttention)
            || (kind is GraphFusion.AdaLnModulate or GraphFusion.GatedResidual && ServesDitModulation);

        /// <summary>
        /// <see cref="GraphFusion.AdaLnModulate"/> where the executor serves it (<see cref="TensorGraphExecutor.ServesDitModulation"/>)
        /// as the retired direct CPU DiT ran it: the RMS norm with its weight in one pass, then each segment's rows
        /// x (1 + scale) + shift in place (DirectOps' row loops). False (the region's ops run) for any other layout.
        /// </summary>
        private bool TryAdaLnModulate(Tensor[] inputs, long[] ip, float[] fp, Tensor[] results)
        {
            if (inputs.Length != 4 || ip == null || ip.Length < 2 || fp == null || fp.Length < 1 || results.Length < 1) return false;
            Tensor w = inputs[1];
            if (w == null || inputs[2] == null || w.ElementCount() == 0 || !w.IsContiguous()) return false;
            long hidden = w.ElementCount();
            using Tensor x = Rows2(inputs[0], hidden), mods = Rows2(inputs[2], hidden);
            if (x == null || mods == null || x.ElementType != DType.Float32 || mods.ElementType != DType.Float32) return false;
            long rows = x.Sizes[0];
            if (!TryRegionSegments(3, rows, mods.Sizes[0], Math.Max(ip[0], ip[1]), out var segments)) return false;
            var y = new Tensor(Allocator, DType.Float32, rows, hidden);
            try
            {
                Ops.RMSNorm(y, x, w, null, fp[0]);
                foreach (var (start, count, column) in segments)
                {
                    using Tensor shift = mods.Narrow(0, column + ip[0], 1), scale = mods.Narrow(0, column + ip[1], 1);
                    TensorSharp.Models.Direct.DirectOps.ModulateRowsOnHost(y, y, shift, scale, start, count);
                }
            }
            catch { y.Dispose(); throw; }
            results[0] = y;
            DitModulationsServedForTests++;
            return true;
        }

        /// <summary>
        /// <see cref="GraphFusion.GatedResidual"/> without a bias or a delta where the executor serves it
        /// (<see cref="TensorGraphExecutor.ServesDitModulation"/>) with the retired direct CPU DiT's arithmetic (residual + mm *
        /// gate, its in-place gated add written to a new tensor): one pass over each segment's rows. False (the region's ops
        /// run) for any other form.
        /// </summary>
        private bool TryGatedResidual(Tensor[] inputs, long[] ip, Tensor[] results)
        {
            if (inputs.Length != 4 || ip == null || ip.Length < 1 || (ip.Length > 1 && ip[1] != 0) || (ip.Length > 2 && ip[2] != 0)
                || results.Length < 1 || inputs[0] == null)
                return false;
            long hidden = inputs[0].Sizes[^1];
            using Tensor residual = Rows2(inputs[0], hidden), mm = Rows2(inputs[1], hidden), mods = Rows2(inputs[2], hidden);
            if (residual == null || mm == null || mods == null || mm.Sizes[0] != residual.Sizes[0]
                || residual.ElementType != DType.Float32 || mm.ElementType != DType.Float32 || mods.ElementType != DType.Float32)
                return false;
            long rows = residual.Sizes[0];
            if (!TryRegionSegments(3, rows, mods.Sizes[0], ip[0], out var segments)) return false;
            var y = new Tensor(Allocator, DType.Float32, rows, hidden);
            try
            {
                foreach (var (start, count, column) in segments)
                {
                    using Tensor gate = mods.Narrow(0, column + ip[0], 1);
                    TensorSharp.Models.Direct.DirectOps.GateAddRowsOnHost(y, residual, mm, gate, start, count);
                }
            }
            catch { y.Dispose(); throw; }
            results[0] = y;
            DitModulationsServedForTests++;
            return true;
        }

        /// <summary>
        /// A diffusion transformer's full attention (<see cref="GraphFusion.DitAttention"/>, served where
        /// <see cref="TensorGraphExecutor.ServesDitAttention"/>, and <see cref="GraphFusion.FullAttention"/> always: the direct CPU
        /// engines' attention, F32 throughout) as the retired direct CPU DiT ran it: the blocked F32 attention
        /// of DirectOps (each head's query blocks against all keys in an L2-sized scratch, the output written in place). The
        /// region's ops (F16 K and V, then the generic attention) held every head's scores of the whole sequence at once.
        /// </summary>
        private bool TryDitAttention(Tensor[] inputs, float[] fp, Tensor[] results)
        {
            if (inputs.Length != 3 || fp == null || fp.Length < 1 || results.Length < 1) return false;
            // [rows, heads, d] each, as a view (an operand may arrive with ggml's leading unit extent).
            using Tensor q = Rows3(inputs[0]), k = Rows3(inputs[1]), v = Rows3(inputs[2]);
            if (q == null || k == null || v == null || q.ElementType != DType.Float32 || k.ElementType != DType.Float32 || v.ElementType != DType.Float32)
                return false;
            long sq = q.Sizes[0], sk = k.Sizes[0], heads = q.Sizes[1], hd = q.Sizes[2];
            if (k.Sizes[1] != heads || v.Sizes[1] != heads || k.Sizes[2] != hd || v.Sizes[2] != hd || v.Sizes[0] != sk
                || sq > int.MaxValue || sk > int.MaxValue || heads > int.MaxValue || hd > int.MaxValue)
                return false;
            Tensor Flat(Tensor t, long rows)
            {
                Tensor c = t.IsContiguous() ? t.CopyRef() : Ops.NewContiguous(t);
                try { return c.View(rows, heads * hd); }
                finally { c.Dispose(); }
            }
            using Tensor qf = Flat(q, sq), kf = Flat(k, sk), vf = Flat(v, sk);
            var output = new Tensor(Allocator, DType.Float32, sq, heads * hd);
            if (!TensorSharp.Models.Direct.DirectOps.TryCpuBlockedAttention(output, qf, kf, vf, null, (int)heads, (int)sq, (int)sk, (int)hd, fp[0]))
            {
                output.Dispose();
                return false;
            }
            results[0] = output;
            DitAttentionsServedForTests++;
            return true;
        }

        /// <summary>
        /// <see cref="GraphFusion.FullAttention"/> as the direct CPU engines attended (DirectOps' blocked F32 attention, through
        /// <see cref="TryDitAttention"/>), then its head gates where the region has them, as those engines gated each head's rows
        /// (sigmoid, the scale, a broadcast multiply in place).
        /// </summary>
        private bool TryFullAttention(Tensor[] inputs, float[] fp, Tensor[] results)
        {
            if (inputs.Length is not (3 or 4) || fp == null || fp.Length < (inputs.Length == 4 ? 2 : 1)) return false;
            Tensor logits = inputs.Length == 4 ? inputs[3] : null;
            if (!TryDitAttention(inputs[..3], fp, results)) return false;
            if (logits != null)
            {
                Tensor output = results[0];
                long rows = output.Sizes[0], heads = logits.ElementCount() / Math.Max(1, rows);
                using Tensor l2 = Contiguous2(logits, rows, heads);
                GateHeads(output, l2, (int)heads, fp[1], contiguousGates: false);
            }
            return true;
        }

        private static Tensor Contiguous2(Tensor t, long rows, long cols)
        {
            Tensor c = t.IsContiguous() ? t.CopyRef() : Ops.NewContiguous(t);
            try { return c.View(rows, cols); }
            finally { c.Dispose(); }
        }

        /// <summary>Tokens of a fused region below which it runs on the calling thread.</summary>
        private const int ParallelRegionRows = 8;

        /// <summary>
        /// The recurrent mixer's regions as single host passes (their semantics are MlxNative.GraphKernels'): a decode step's
        /// dozen small ops per Gated DeltaNet layer, each a few microseconds of dispatch on the host, become two loops.
        /// </summary>
        protected override unsafe bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            // A diffusion transformer's unmerged LoRA (its weights are leaves of their own, not tensors).
            if (kind == GraphFusion.LoraLinear) return TryLoraLinearInPlace(inputs, weights, ip, results);
            if (kind == GraphFusion.DitAttention) return TryDitAttention(inputs, fp, results);
            if (kind == GraphFusion.FullAttention) return TryFullAttention(inputs, fp, results);
            if (kind == GraphFusion.AdaLnModulate) return TryAdaLnModulate(inputs, ip, fp, results);
            if (kind == GraphFusion.GatedResidual) return TryGatedResidual(inputs, ip, results);
            foreach (Tensor t in inputs)
                if (t == null || t.ElementType != DType.Float32) return false;
            float eps = fp != null && fp.Length > 0 ? fp[0] : 1e-6f;
            switch (kind)
            {
                case GraphFusion.PairRotary:
                    return PairRotary(inputs[0], inputs[1], inputs[2], results);
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
        /// group, SIMD dots, the sequence split over cores with an online softmax), reading the cache window in place. On ARM64
        /// also a row that sees one run of keys past a floor (a sliding-window layer past its window: the per-op path's
        /// WindowDecodeOnHost does the same), and every other shape - prompt chunks, verify windows, a ring's wrapped band -
        /// through <see cref="HostFlashAttention"/>. The rest take the interpreter's attention.
        /// </summary>
        protected override unsafe Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow, bool promptChunk)
        {
            long heads = q.Sizes[0], d = q.Sizes[2], kvHeads = k.Sizes[0], keys = k.Sizes[1];
            // A padded decode window (its unwritten tail masked) runs its pattern over the written keys only.
            bool trimmed = mask.Pattern == MaskPattern.Explicit && mask.Trimmed != MaskPattern.Explicit && mask.Keys > 0;
            MaskPattern pattern = trimmed ? mask.Trimmed : mask.Pattern;
            int floor = trimmed ? mask.TrimmedFloor : mask.Floor;
            bool half = k.ElementType == DType.Float16 && v.ElementType == DType.Float16;
            bool layout =v.Sizes[2] == d && k.Sizes[2] == d && kvHeads > 0 && heads % kvHeads == 0
                && q.ElementType == DType.Float32 && (half || (k.ElementType == DType.Float32 && v.ElementType == DType.Float32))
                && k.Strides[2] == 1 && k.Strides[1] == d && v.Strides[2] == 1 && v.Strides[1] == d
                && k.Strides[0] == v.Strides[0] && k.Strides[0] % d == 0 && v.Sizes[1] == keys;
            // One row over the run of keys [floor, end): the decode kernel (past a floor on ARM64 only).
            bool run = q.Sizes[1] == 1 && (pattern == MaskPattern.None || pattern == MaskPattern.Causal)
                && (floor == 0 || HostFlashAttention.Supported);
            if (layout && !run && HostFlashAttention.Supported)
            {
                // A few rows (a decode step of several sequences, a verify window): each row whose keys are one run takes the
                // decode kernel over that run, as it would alone, so its result does not depend on what else the step carried.
                if (q.Sizes[1] <= HostWindowRows
                    && TryRowsAsDecodes(q, k, v, mask, scale, trimmed, pattern, floor, half, out Tensor rows))
                    return rows;
                if (TryHostFlashAttention(q, k, v, mask, scale, trimmed, pattern, floor, half, out Tensor tiled))
                    return tiled;
            }
            if (!run || !layout)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow, promptChunk);
            if (trimmed) keys = mask.Keys;
            else if (floor > 0 && mask.Keys > 0) keys = Math.Min(keys, mask.Keys);
            if (floor >= keys)
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow, promptChunk);
            Tensor packed = q.IsContiguous() ? null : Ops.NewContiguous(q);
            try
            {
                var result = new Tensor(Allocator, DType.Float32, 1, heads, d);
                int stride = checked((int)(k.Strides[0] / d));
                float* qp = TensorComputePrimitives.GetFloatPointer(packed ?? q), rp = TensorComputePrimitives.GetFloatPointer(result);
                long skip = (long)floor * d;
                if (half)
                    ModelBase.AttentionDecodeGroupedF16(qp, TensorComputePrimitives.GetHalfPointer(k) + skip, TensorComputePrimitives.GetHalfPointer(v) + skip,
                        rp, (int)heads, (int)kvHeads, (int)d, stride, checked((int)(keys - floor)), scale);
                else
                    ModelBase.AttentionDecodeGrouped(qp, TensorComputePrimitives.GetFloatPointer(k) + skip, TensorComputePrimitives.GetFloatPointer(v) + skip,
                        rp, (int)heads, (int)kvHeads, (int)d, stride, checked((int)(keys - floor)), scale, Environment.ProcessorCount);
                return result;
            }
            finally
            {
                packed?.Dispose();
            }
        }

        /// <summary>Rows of an attention node from which <see cref="HostFlashAttention"/> takes over from row-by-row decodes.</summary>
        private const int HostWindowRows = 16;

        /// <summary>
        /// Each row of a few-row node through the decode kernel over its own run of keys - [floor, its causal end) for a
        /// causal or open mask, the visible span of its row of an explicit mask when that span is one run of zeros (a row
        /// with a wrapped band goes through <see cref="HostFlashAttention"/> alone, as it would in a step of its own). False
        /// when the mask cannot be read.
        /// </summary>
        private unsafe bool TryRowsAsDecodes(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale, bool trimmed,
            MaskPattern pattern, int floor, bool half, out Tensor result)
        {
            result = null;
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2], kvHeads = (int)k.Sizes[0], nkv = (int)k.Sizes[1];
            int valid = mask.Keys >= 0 ? Math.Min(mask.Keys, nkv) : nkv;
            Tensor m = null;
            if (pattern == MaskPattern.Explicit)
            {
                if (!mask.HasTensor) return false;
                m = mask.Tensor;
                if (m == null || m.DimensionCount != 2 || m.Sizes[0] != nq || m.Sizes[1] < valid || m.Strides[1] != 1
                    || !(m.ElementType == DType.Float32 || m.ElementType == DType.Float16))
                    return false;
            }
            var output = new Tensor(Allocator, DType.Float32, nq, heads, d);
            float* qp = TensorComputePrimitives.GetFloatPointer(q), op = TensorComputePrimitives.GetFloatPointer(output);
            long qHead = q.Strides[0], qRow = q.Strides[1];
            int stride = checked((int)(k.Strides[0] / d));
            float[] rowQ = new float[heads * d];
            fixed (float* rq = rowQ)
            {
                for (int r = 0; r < nq; r++)
                {
                    int first, end;
                    if (m == null)
                    {
                        first = floor;
                        end = pattern == MaskPattern.Causal ? Math.Clamp(valid - nq + r + 1, floor, valid) : valid;
                    }
                    else if (!VisibleRun(m, r, valid, out first, out end))
                    {
                        // A row whose keys are not one run: the tiled kernel on that row alone.
                        using Tensor qr = q.Narrow(1, r, 1);
                        using Tensor mr = m.Narrow(0, r, 1);
                        if (!TryOneRowTiled(qr, k, v, mr, scale, valid, half, op + (long)r * heads * d))
                        {
                            output.Dispose();
                            return false;
                        }
                        continue;
                    }
                    if (end <= first)
                    {
                        output.Dispose();
                        return false;
                    }
                    for (int h = 0; h < heads; h++)
                        Buffer.MemoryCopy(qp + h * qHead + r * qRow, rq + (long)h * d, d * sizeof(float), d * sizeof(float));
                    long skip = (long)first * d;
                    float* rp = op + (long)r * heads * d;
                    if (half)
                        ModelBase.AttentionDecodeGroupedF16(rq, TensorComputePrimitives.GetHalfPointer(k) + skip, TensorComputePrimitives.GetHalfPointer(v) + skip,
                            rp, heads, kvHeads, d, stride, end - first, scale);
                    else
                        ModelBase.AttentionDecodeGrouped(rq, TensorComputePrimitives.GetFloatPointer(k) + skip, TensorComputePrimitives.GetFloatPointer(v) + skip,
                            rp, heads, kvHeads, d, stride, end - first, scale, Environment.ProcessorCount);
                }
            }
            result = output;
            return true;
        }

        /// <summary>The keys row <paramref name="r"/> of an explicit mask sees, when they are one run [first, end) of zeros
        /// with -inf everywhere else in [0, valid).</summary>
        private static unsafe bool VisibleRun(Tensor m, int r, int valid, out int first, out int end)
        {
            first = end = 0;
            bool halfMask = m.ElementType == DType.Float16;
            byte* row = (byte*)(halfMask ? (void*)TensorComputePrimitives.GetHalfPointer(m) : TensorComputePrimitives.GetFloatPointer(m))
                + (long)r * m.Strides[0] * (halfMask ? 2 : 4);
            float At(int j) => halfMask ? (float)BitConverter.UInt16BitsToHalf(((ushort*)row)[j]) : ((float*)row)[j];
            int j0 = 0;
            while (j0 < valid && float.IsNegativeInfinity(At(j0))) j0++;
            int j1 = j0;
            while (j1 < valid && At(j1) == 0f) j1++;
            for (int j = j1; j < valid; j++)
                if (!float.IsNegativeInfinity(At(j))) return false;
            first = j0;
            end = j1;
            return end > first;
        }

        /// <summary>One row through <see cref="HostFlashAttention"/> with its additive mask row, into [heads, d] at
        /// <paramref name="dst"/>.</summary>
        private unsafe bool TryOneRowTiled(Tensor qr, Tensor k, Tensor v, Tensor mr, float scale, int valid, bool half, float* dst)
        {
            int heads = (int)qr.Sizes[0], d = (int)qr.Sizes[2], kvHeads = (int)k.Sizes[0];
            using Tensor packed = Ops.NewContiguous(qr);
            void* kp = half ? TensorComputePrimitives.GetHalfPointer(k) : TensorComputePrimitives.GetFloatPointer(k);
            void* vp = half ? TensorComputePrimitives.GetHalfPointer(v) : TensorComputePrimitives.GetFloatPointer(v);
            void* mp = mr.ElementType == DType.Float16 ? TensorComputePrimitives.GetHalfPointer(mr) : TensorComputePrimitives.GetFloatPointer(mr);
            HostFlashAttention.Run(TensorComputePrimitives.GetFloatPointer(packed), d, kp, vp, k.Strides[0], half, heads, kvHeads, 1, d, scale,
                HostFlashAttention.MaskKind.Additive, 0, valid, mp, mr.Strides[0], mr.ElementType == DType.Float16, dst);
            return true;
        }

        /// <summary>
        /// Every other attention node on ARM64 - prompt chunks, verify windows, and decode rows whose mask has a floor or a band
        /// (sliding-window layers past their window, ring caches) - through <see cref="HostFlashAttention"/>: tiled over the
        /// cache in place with an online softmax, where the interpreter's attention widens the K/V to F32, repeats them per
        /// query head and materializes [heads, rows, keys] scores. False (the interpreter runs) for a mask it cannot read.
        /// </summary>
        private unsafe bool TryHostFlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale, bool trimmed,
            MaskPattern pattern, int floor, bool half, out Tensor result)
        {
            result = null;
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2], kvHeads = (int)k.Sizes[0], nkv = (int)k.Sizes[1];
            int valid = mask.Keys >= 0 ? Math.Min(mask.Keys, nkv) : nkv;
            HostFlashAttention.MaskKind kind;
            Tensor m = null;
            if (pattern == MaskPattern.None) kind = HostFlashAttention.MaskKind.None;
            else if (pattern == MaskPattern.Causal) kind = HostFlashAttention.MaskKind.Causal;
            else
            {
                if (!mask.HasTensor) return false;
                m = mask.Tensor;
                if (m == null || m.DimensionCount != 2 || m.Sizes[0] != nq || m.Sizes[1] < valid || m.Strides[1] != 1
                    || !(m.ElementType == DType.Float32 || m.ElementType == DType.Float16))
                    return false;
                kind = HostFlashAttention.MaskKind.Additive;
                floor = 0;
            }
            if (!trimmed && pattern != MaskPattern.Explicit && mask.Keys < 0) valid = nkv;
            Tensor packed = q.IsContiguous() ? null : Ops.NewContiguous(q);
            try
            {
                var output = new Tensor(Allocator, DType.Float32, nq, heads, d);
                void* kp = half ? TensorComputePrimitives.GetHalfPointer(k) : TensorComputePrimitives.GetFloatPointer(k);
                void* vp = half ? TensorComputePrimitives.GetHalfPointer(v) : TensorComputePrimitives.GetFloatPointer(v);
                void* mp = m == null ? null
                    : m.ElementType == DType.Float16 ? TensorComputePrimitives.GetHalfPointer(m) : TensorComputePrimitives.GetFloatPointer(m);
                HostFlashAttention.Run(TensorComputePrimitives.GetFloatPointer(packed ?? q), (long)nq * d, kp, vp, k.Strides[0], half,
                    heads, kvHeads, nq, d, scale, kind, floor, valid, mp, m?.Strides[0] ?? 0, m?.ElementType == DType.Float16,
                    TensorComputePrimitives.GetFloatPointer(output));
                result = output;
                return true;
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

        /// <summary>
        /// ggml_ssm_scan (<see cref="TensorGraphExecutor.SsmScan"/>) on host memory: each (sequence, head) state updated in
        /// place in its newest slot, a SIMD pass over the state row per (head dim, token) that decays, adds B x dt and dots C
        /// in one read, and the pairs in parallel once the step has enough work (as the removed legacy Nemotron-H model's managed Mamba-2 step ran).
        /// </summary>
        protected override unsafe Tensor SsmScan(Tensor state, Tensor x, Tensor dt, Tensor a, Tensor b, Tensor c, int[] ids, int keep)
        {
            if (!state.IsContiguous() || !x.IsContiguous() || !dt.IsContiguous() || !a.IsContiguous() || !b.IsContiguous() || !c.IsContiguous())
                return base.SsmScan(state, x, dt, a, b, c, ids, keep);
            int ns = (int)x.Sizes[0], nt = (int)x.Sizes[1], nh = (int)x.Sizes[2], hd = (int)x.Sizes[3];
            int ng = (int)b.Sizes[2], dstate = (int)b.Sizes[3];
            int slots = Math.Max(1, keep);
            long yFloats = (long)ns * nt * nh * hd, perHead = (long)hd * dstate, perState = perHead * nh, perSlot = perState * ns;
            var result = new Tensor(Allocator, DType.Float32, yFloats + perSlot * slots);
            float* outp = TensorComputePrimitives.GetFloatPointer(result);
            // A slot older than the step's tokens stays zero, as ggml leaves it.
            if (slots > nt)
                new Span<float>(outp + yFloats + perSlot * nt, checked((int)(perSlot * (slots - nt)))).Clear();
            long xa = (long)TensorComputePrimitives.GetFloatPointer(x), da = (long)TensorComputePrimitives.GetFloatPointer(dt);
            long aa = (long)TensorComputePrimitives.GetFloatPointer(a), ba = (long)TensorComputePrimitives.GetFloatPointer(b);
            long ca = (long)TensorComputePrimitives.GetFloatPointer(c), sa = (long)TensorComputePrimitives.GetFloatPointer(state);
            long oa = (long)outp;
            int headsPerGroup = nh / ng;

            void Pair(int pair)
            {
                int seq = pair / nh, h = pair % nh, g = h / headsPerGroup;
                float* s = (float*)oa + yFloats + (long)seq * perState + h * perHead;
                float* from = (float*)sa + (long)ids[seq] * perState + h * perHead;
                if (from != s)
                    Buffer.MemoryCopy(from, s, perHead * sizeof(float), perHead * sizeof(float));
                float A = ((float*)aa)[h];
                for (int t = 0; t < nt; t++)
                {
                    long row = (long)seq * nt + t;
                    float dtSoft = TensorComputePrimitives.Softplus(((float*)da)[row * nh + h]);
                    float dA = MathF.Exp(dtSoft * A);
                    float* bg = (float*)ba + (row * ng + g) * dstate, cg = (float*)ca + (row * ng + g) * dstate;
                    float* xh = (float*)xa + (row * nh + h) * hd, y = (float*)oa + (row * nh + h) * hd;
                    for (int i = 0; i < hd; i++)
                        y[i] = ScanRow(s + (long)i * dstate, bg, cg, dA, xh[i] * dtSoft, dstate);
                    int slot = nt - 1 - t;
                    if (slot >= 1 && slot < slots)
                        Buffer.MemoryCopy(s, (float*)oa + yFloats + slot * perSlot + (long)seq * perState + h * perHead,
                            perHead * sizeof(float), perHead * sizeof(float));
                }
            }

            int pairs = ns * nh;
            if (pairs > 1 && perHead * nt * pairs >= ParallelRecurrenceWork)
                System.Threading.Tasks.Parallel.For(0, pairs, Pair);
            else
                for (int pair = 0; pair < pairs; pair++) Pair(pair);
            return result;
        }

        /// <summary>One state row of the selective scan: row = row * dA + b * xdt, returning its dot with c.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float ScanRow(float* row, float* b, float* c, float dA, float xdt, int n)
        {
            int w = System.Numerics.Vector<float>.Count, i = 0;
            var acc = System.Numerics.Vector<float>.Zero;
            var decay = new System.Numerics.Vector<float>(dA);
            var input = new System.Numerics.Vector<float>(xdt);
            for (; i <= n - w; i += w)
            {
                var r = TensorComputePrimitives.LoadVector(row + i) * decay + TensorComputePrimitives.LoadVector(b + i) * input;
                TensorComputePrimitives.StoreVector(row + i, r);
                acc += r * TensorComputePrimitives.LoadVector(c + i);
            }
            float sum = System.Numerics.Vector.Sum(acc);
            for (; i < n; i++)
            {
                row[i] = row[i] * dA + b[i] * xdt;
                sum += row[i] * c[i];
            }
            return sum;
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
            {
                // On the worker pool (whose threads would otherwise spin through a ThreadPool loop), channels in chunks.
                long perChunk = Math.Max(1, (ParallelRecurrenceWork * 4) / Math.Max(1, nt * width));
                int chunks = (int)((rows + perChunk - 1) / perChunk);
                CpuParallel.For(chunks, c =>
                {
                    long i1 = Math.Min(rows, (c + 1) * perChunk);
                    for (long i = c * perChunk; i < i1; i++) Channel(i);
                });
            }
            else
                for (long i = 0; i < rows; i++) Channel(i);
            return result;
        }

        /// <summary>Elements a parallel chunk of a broadcast binary op takes.</summary>
        private const long BroadcastChunkElements = 1 << 15;

        /// <summary>
        /// A binary op whose b repeats over a (the attention output gate per head, the short convolutions' gates), F32 and
        /// contiguous: one pass over the rows on the worker pool, b read in place - where the interpreter's expanded view ran
        /// TensorApplyCPU's generic strided loop on one thread (12 % of an LFM3 prompt step on the managed lane). Each element
        /// is the same single IEEE operation (a op b), so the result is the same bits.
        /// </summary>
        protected override unsafe bool TryBinaryBroadcast(GraphOp kind, Tensor result, Tensor a, Tensor b, long[] na, long[] nb)
        {
            if (kind is not (GraphOp.Add or GraphOp.Sub or GraphOp.Mul or GraphOp.Div)) return false;
            if (result.ElementType != DType.Float32 || a.ElementType != DType.Float32 || b.ElementType != DType.Float32
                || !result.IsContiguous() || !a.IsContiguous() || !b.IsContiguous() || na.Length != 4 || nb.Length != 4)
                return false;
            long n0 = na[0], n1 = na[1], n2 = na[2], rows = na[1] * na[2] * na[3];
            long b1 = nb[0], b2 = nb[0] * nb[1], b3 = nb[0] * nb[1] * nb[2];   // b's strides in its own shape
            bool inner = nb[0] == n0;                                           // b varies along the row
            long ra = (long)TensorComputePrimitives.GetFloatPointer(result), aa = (long)TensorComputePrimitives.GetFloatPointer(a);
            long ba = (long)TensorComputePrimitives.GetFloatPointer(b);
            long rowsPerChunk = Math.Max(1, BroadcastChunkElements / Math.Max(1, n0));
            int chunks = (int)Math.Min(int.MaxValue, (rows + rowsPerChunk - 1) / rowsPerChunk);

            void Chunk(int c)
            {
                long r0 = c * rowsPerChunk, r1 = Math.Min(rows, r0 + rowsPerChunk);
                for (long row = r0; row < r1; row++)
                {
                    long i1 = row % n1, i2 = row / n1 % n2, i3 = row / (n1 * n2);
                    long boff = (nb[1] == 1 ? 0 : i1 * b1) + (nb[2] == 1 ? 0 : i2 * b2) + (nb[3] == 1 ? 0 : i3 * b3);
                    float* r = (float*)ra + row * n0, x = (float*)aa + row * n0, y = (float*)ba + boff;
                    if (inner) BinaryRow(kind, r, x, y, n0);
                    else BinaryRowScalar(kind, r, x, *y, n0);
                }
            }

            if (chunks > 1) CpuParallel.For(chunks, Chunk);
            else Chunk(0);
            return true;
        }

        private static unsafe void BinaryRow(GraphOp kind, float* r, float* x, float* y, long n)
        {
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> u = Vector128.Load(x + i), w = Vector128.Load(y + i);
                Vector128.Store(kind switch { GraphOp.Add => u + w, GraphOp.Sub => u - w, GraphOp.Mul => u * w, _ => u / w }, r + i);
            }
            for (; i < n; i++)
                r[i] = kind switch { GraphOp.Add => x[i] + y[i], GraphOp.Sub => x[i] - y[i], GraphOp.Mul => x[i] * y[i], _ => x[i] / y[i] };
        }

        private static unsafe void BinaryRowScalar(GraphOp kind, float* r, float* x, float y, long n)
        {
            Vector128<float> w = Vector128.Create(y);
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                Vector128<float> u = Vector128.Load(x + i);
                Vector128.Store(kind switch { GraphOp.Add => u + w, GraphOp.Sub => u - w, GraphOp.Mul => u * w, _ => u / w }, r + i);
            }
            for (; i < n; i++)
                r[i] = kind switch { GraphOp.Add => x[i] + y, GraphOp.Sub => x[i] - y, GraphOp.Mul => x[i] * y, _ => x[i] / y };
        }

        /// <summary>
        /// x [rows, Ne0] times the weight's rows. A program whose products keep their activations in F32
        /// (<see cref="TensorGraphExecutor.FloatProducts"/>: the F32 reference tower of the Muse-Glimmer vision tests) multiplies
        /// the weight's values (ggml's dequantization; code * d in F32 for q8_0) as F32 rows, the arithmetic of a per-op path that
        /// dequantized its weights at load, at about 2.2x the time. The managed kernels' own q8_0 product is an integer dot over
        /// quantized activations: 2.2e-2 from F32 copies over the Muse-Glimmer tower's synthetic Q8_0 fixture, 1 - cosine 9e-4 on
        /// the real mmproj's rows.
        /// </summary>
        protected override unsafe Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            if (weight.Data == IntPtr.Zero)
                throw new InvalidOperationException("a quantized weight's host bytes were released");
            int rows = (int)x.Sizes[0];
            SetStepWidth(rows);
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            // The direct engines' CPU linear (DirectProducts): the managed quantized product over the stored blocks, an F32
            // weight (a LoRA factor) on the SIMD SGEMM over its values, as DirectLinear held them.
            if (DirectProducts && (weight.ExactProducts || weight.Format == GraphFormat.F32) && weight.Ne2 <= 1)
            {
                TensorSharp.Models.Direct.DirectOps.CpuGemmABt(x, DirectF32(weight), result, 1f, 0f);
                return result;
            }
            if (FloatProducts && weight.Ne2 <= 1)
            {
                using var values = new Tensor(Allocator, DType.Float32, weight.Ne1, weight.Ne0);
                DequantizeValues(weight, (IntPtr)TensorComputePrimitives.GetFloatPointer(values));
                DenseProduct(result, x, values);
                return result;
            }
            ManagedQuantizedOps.AddmmQuantizedToFloat32(weight.Format, weight.Data, weight.Ne0, weight.Ne1,
                TensorComputePrimitives.GetFloatPointer(x), (int)weight.Ne0, rows,
                TensorComputePrimitives.GetFloatPointer(result), (int)weight.Ne1);
            return result;
        }

        /// <summary>
        /// Every routed expert's product in one batched dispatch (<see cref="ManagedQuantizedOps.TryAddmmQuantizedBatch"/>: the
        /// activations quantized once, the experts cut into column slices sized from the batch's work so a hot expert is not the
        /// fork/join's straggler), where the per-expert loop had taken a pool fork/join per expert, most of them a few rows wide
        /// (a 256-row DiffusionGemma canvas routes about 16 rows to each of 128 experts a layer). Row for row the dense call's
        /// dot products; other formats and F32 products take the loop.
        /// </summary>
        protected override unsafe Tensor ExpertProducts(IGraphExpertWeights set, Tensor rows, int[] offsets, RunState run)
        {
            int experts = offsets.Length - 1, routes = (int)rows.Sizes[0];
            IGraphWeight first = null;
            int active = 0;
            for (int e = 0; e < experts; e++)
            {
                if (offsets[e + 1] == offsets[e]) continue;
                IGraphWeight w = set.Expert(e);
                first ??= w;
                if (w.Data == IntPtr.Zero || w.Format != first.Format || w.Ne0 != first.Ne0 || w.Ne1 != first.Ne1)
                    return base.ExpertProducts(set, rows, offsets, run);
                active++;
            }
            if (FloatProducts || first == null || active < 2 || !rows.IsContiguous() || rows.Sizes[1] != first.Ne0)
                return base.ExpertProducts(set, rows, offsets, run);
            int inDim = (int)first.Ne0, outDim = (int)first.Ne1;
            SetStepWidth(routes);
            var result = new Tensor(Allocator, DType.Float32, routes, outDim);
            float* x = TensorComputePrimitives.GetFloatPointer(rows), y = TensorComputePrimitives.GetFloatPointer(result);
            long rowBytes = ManagedQuantizedOps.RowSize(first.Format, inDim);
            long perItem = Math.Max(1, (long)routes * outDim / Math.Max(1, CpuWorkerPool.Shared.ThreadCount * 4));
            var jobs = new System.Collections.Generic.List<ManagedQuantizedOps.QuantMatMulJob>(active * 2);
            for (int e = 0; e < experts; e++)
            {
                int count = offsets[e + 1] - offsets[e];
                if (count == 0) continue;
                long work = (long)count * outDim;
                int pieces = (int)Math.Clamp((work + perItem - 1) / perItem, 1, Math.Max(1, outDim / 64));
                int colsPer = (outDim + pieces - 1) / pieces;
                IntPtr weights = set.Expert(e).Data;
                for (int c0 = 0; c0 < outDim; c0 += colsPer)
                    jobs.Add(new ManagedQuantizedOps.QuantMatMulJob(weights + (nint)(c0 * rowBytes), (IntPtr)(x + (long)offsets[e] * inDim),
                        (IntPtr)(y + (long)offsets[e] * outDim + c0), Math.Min(colsPer, outDim - c0), count, outDim));
            }
            if (ManagedQuantizedOps.TryAddmmQuantizedBatch(first.Format, inDim, inDim, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(jobs)))
                return result;
            result.Dispose();
            return base.ExpertProducts(set, rows, offsets, run);
        }

        // The F32 weights of direct products as tensors (DirectLinear's own copies), by host address; Invalidate drops one.
        private readonly System.Collections.Generic.Dictionary<IntPtr, Tensor> _directF32 = new();

        private unsafe Tensor DirectF32(IGraphWeight weight)
        {
            lock (_directF32)
            {
                if (_directF32.TryGetValue(weight.Data, out Tensor cached)) return cached;
                if (weight.Format != GraphFormat.F32 || weight.Bytes != weight.Ne0 * weight.Ne1 * sizeof(float))
                    throw new NotSupportedException($"an exact product reads an F32 weight, not format {weight.Format}");
                var values = new Tensor(Allocator, DType.Float32, weight.Ne1, weight.Ne0);
                Buffer.MemoryCopy((void*)weight.Data, TensorComputePrimitives.GetFloatPointer(values), weight.Bytes, weight.Bytes);
                return _directF32[weight.Data] = values;
            }
        }

        /// <summary>Drops the F32 copy a direct product made of the weight at <paramref name="data"/>.</summary>
        public override void Invalidate(IntPtr data)
        {
            base.Invalidate(data);
            lock (_directF32)
                if (_directF32.Remove(data, out Tensor t)) t.Dispose();
        }

        /// <summary>Rows of a weight one parallel chunk of <see cref="DequantizeValues"/> takes at least.</summary>
        private const long DequantizeChunkRows = 16;

        /// <summary>
        /// An exact product's weight values (<see cref="TensorGraphExecutor.FloatProducts"/>) as F32 rows, the rows in chunks on
        /// the worker pool. A row's values are its blocks' dequantization whichever thread writes them, so they are the bits one
        /// call over the whole weight writes. That call ran on one thread: about 90 s of every step of the Muse-Glimmer F32
        /// reference on an M5 Pro (eng/validation/MuseGlimmerF32Reference, a dense 30B model's weights each step), with the 17
        /// other pool threads busy 6% of it.
        /// </summary>
        internal static void DequantizeValues(IGraphWeight weight, IntPtr values)
        {
            long ne0 = weight.Ne0, rows = weight.Ne1, rowBytes = weight.Bytes / Math.Max(1, rows);
            int format = weight.Format;
            IntPtr data = weight.Data;
            if (rowBytes * rows != weight.Bytes)
            {
                NativeDequant.DequantizeToFloat32Native(format, data, values, ne0 * rows);
                return;
            }
            CpuParallel.ForRange(rows, DequantizeChunkRows, (from, to) =>
                NativeDequant.DequantizeToFloat32Native(format, data + (nint)(from * rowBytes),
                    values + (nint)(from * ne0 * sizeof(float)), (to - from) * ne0));
        }
    }
}
