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

        /// <summary>The base executor's capabilities, and a decode row's recurrence updates its state in place (see
        /// <see cref="GatedDeltaNet"/>).</summary>
        public override GraphCapabilities Capabilities { get; } = new()
        {
            StridedViews = true, DecodeWindowRows = 1, Replays = false, IndexedCacheWrites = true, InPlaceStateAlias = true,
        };

        /// <summary>CUDA's binary kernels take same-shaped contiguous operands: a broadcast operand is gathered.</summary>
        protected override bool BroadcastsInKernels => false;

        /// <summary>
        /// Many short strided rows (a conv window's update: each channel's last width-1 columns, 6144 rows of 3 floats on
        /// Qwen3.5-0.8B) gather through the head-first kernel, a thread an element. The generic pitched copy runs a block a
        /// row: 124 us a layer, a third of that model's decode GPU time.
        /// </summary>
        protected override void CopyInto(Tensor dst, Tensor src)
        {
            if (!TryGatherShortRows(dst, src)) base.CopyInto(dst, src);
        }

        private static bool TryGatherShortRows(Tensor dst, Tensor src)
        {
            if (src.DimensionCount != 2 || src.ElementType != DType.Float32 || dst.ElementType != DType.Float32
                || !dst.IsContiguous() || dst.ElementCount() != src.ElementCount() || src.Strides[1] != 1
                || src.Storage is not CudaStorage from || dst.Storage is not CudaStorage to || from.DeviceId != to.DeviceId)
                return false;
            long rows = src.Sizes[0], width = src.Sizes[1], pitch = src.Strides[0];
            if (rows < 64 || width > 16 || pitch <= width || pitch > int.MaxValue)
                return false;
            long column = src.StorageOffset % pitch, start = src.StorageOffset - column;
            if (column + width > pitch || start + rows * pitch > src.Storage.ElementCount)
                return false;
            using var cache = new Tensor(new[] { rows, pitch, 1L }, new[] { pitch, 1L, 1L }, src.Storage, start);
            using Tensor result = dst.View(rows, width, 1);
            return CudaFusedOps.TryGatherCircularHeadFirst(result, cache, (int)column, (int)width, (int)pitch);
        }

        /// <summary>The recurrence wrote its result into the leaf already (<see cref="GatedDeltaNet"/>): nothing to copy.</summary>
        protected override bool TryAlias(Tensor result, Tensor leaf, long offsetBytes)
            => ReferenceEquals(result.Storage, leaf.Storage)
               && result.StorageOffset * result.ElementType.Size() == leaf.StorageOffset * leaf.ElementType.Size() + offsetBytes;

        /// <summary>A prompt chunk's or a decode row's K/V rows into the head-first cache (F32 or F16) with the
        /// per-op path's kernel.</summary>
        protected override bool TryWriteRows(Tensor dst, Tensor src, int first, int count)
        {
            if (dst.DimensionCount is < 2 or > 3 || !dst.IsContiguous()) return false;
            using Tensor cache = HeadFirst(dst);
            using Tensor rows = src.View(src.Sizes[^3], src.Sizes[^2], src.Sizes[^1]);
            Tensor contiguous = rows.IsContiguous() ? rows.CopyRef() : Ops.NewContiguous(rows);
            using (contiguous)
                return CudaFusedOps.TryCopyHeadFirstToCache(cache, contiguous, first, count, (int)cache.Sizes[1], false);
        }

        /// <summary>A head-first cache or window [heads, cells, dim] as the kernels take it: a single head's binds as
        /// [cells, dim] (a leaf's trailing unit extents are dropped), so it is viewed with its head again.</summary>
        private static Tensor HeadFirst(Tensor t) => t.DimensionCount == 2 ? t.View(1, t.Sizes[0], t.Sizes[1]) : t.CopyRef();

        /// <summary>
        /// Attention over cache windows with the per-op path's kernels: a decode row runs the GQA decode kernel
        /// on the head-first cache in place (F32 or F16); a prompt chunk expands the window's KV heads, runs the
        /// half-operand GEMMs of the per-op prefill with the causal mask, and the softmax. Anything else (an
        /// explicit mask, an operand that is not a cache window) takes the generic path.
        /// </summary>
        protected override Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            if (kWindow == null || vWindow == null || mask.Pattern == MaskPattern.Explicit || q.ElementType != DType.Float32
                || q.DimensionCount != 3 || k.DimensionCount is < 2 or > 3 || v.DimensionCount is < 2 or > 3
                || kWindow.Cache.DimensionCount is < 2 or > 3 || vWindow.Cache.DimensionCount is < 2 or > 3
                || k.Sizes[^1] != v.Sizes[^1])
                return base.FlashAttention(q, k, v, mask, scale, kWindow, vWindow);
            int heads = (int)q.Sizes[0], nq = (int)q.Sizes[1], d = (int)q.Sizes[2];
            // A single KV head's window and cache bind without their head extent.
            int kvHeads = k.DimensionCount == 3 ? (int)k.Sizes[0] : 1, len = (int)k.Sizes[^2], floor = mask.Floor;
            using Tensor kc = HeadFirst(kWindow.Cache), vc = HeadFirst(vWindow.Cache);
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

        protected override bool Fuses(GraphFusion kind) => kind switch
        {
            GraphFusion.DeltaNetPrep or GraphFusion.GatedRmsNorm => Kernels != null,
            GraphFusion.RoutedExperts => ((CudaAllocator)Allocator).Kernels != null,
            _ => false,
        };

        /// <summary>The delta rule's prep and the gated norm after it as one kernel each; routed experts with the per-op
        /// path's MoE kernels.</summary>
        protected override bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            if (kind == GraphFusion.RoutedExperts) return TryRoutedExperts(inputs, weights, ip, results);
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
                    if (ip != null && ip.Length > 0 && ip[0] != 0) return false;   // the kernel gates with silu only
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

        private static void RequireDevice(Tensor t, int device, string role, int index)
        {
            if (t.Storage is CudaStorage s && s.DeviceId != device)
                throw new InvalidOperationException(
                    $"CUDA graph kernel {role} {index} [{string.Join(", ", t.Sizes.ToArray())}] lives on device {s.DeviceId}; the executor runs on device {device}.");
        }

        /// <summary>Make the inputs and the output device-current, launch on the allocator's stream, mark the output written.</summary>
        private void Launch(Tensor[] inputs, Tensor output, Action<CudaGraphKernels, IntPtr> launch)
            => Launch(inputs, new[] { output }, launch);

        private void Launch(Tensor[] inputs, Tensor[] outputs, Action<CudaGraphKernels, IntPtr> launch)
        {
            var cuda = (CudaAllocator)Allocator;
            // A kernel reads its operands by device pointer in this executor's context: an operand on another GPU (a
            // layer-split stage reading another stage's memory) would be an illegal access, reported only later.
            for (int i = 0; i < inputs.Length; i++) RequireDevice(inputs[i], cuda.DeviceId, "input", i);
            for (int i = 0; i < outputs.Length; i++) RequireDevice(outputs[i], cuda.DeviceId, "output", i);
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

            long rowFloats = (long)sequences * tokens * heads * s;
            long total = rowFloats + (long)slots * sequences * heads * s * s;
            var cuda = (CudaAllocator)Allocator;
            // A decode row whose state is the tail of a [row outputs | state] buffer (a slot's backing): the result goes
            // there, the new state over the old (each warp reads its state row before writing it back), and the alias
            // that follows finds it in place - a 1 MB state copy per layer and token otherwise (Qwen3.5-0.8B: 32% of
            // the decode GPU time).
            bool inPlace = slots == 1 && tokens == 1 && sequences == 1 && state.Storage is CudaStorage held
                && state.StorageOffset == rowFloats && held.ElementCount == total && held.DeviceId == cuda.DeviceId;
            Tensor result = inPlace
                ? new Tensor(new[] { total }, new[] { 1L }, state.Storage, 0)
                : new Tensor(Allocator, DType.Float32, total);
            foreach (Tensor t in inputs) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            var output = (CudaStorage)result.Storage;
            output.EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            cuda.Context.MakeCurrent();
            // A chunk of rows reads each token's q and k once per four state rows; a decode row keeps the one-row kernel.
            if (tokens > 1)
                kernels.LaunchGatedDeltaNetRows4(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(0),
                    s, heads, keyHeads, tokens, sequences, slots, kda, cuda.Stream.Handle);
            else
                kernels.LaunchGatedDeltaNet(P(q), P(k), P(v), P(g), P(beta), P(state), output.DevicePtrAtElement(0),
                    s, heads, keyHeads, tokens, sequences, slots, kda, cuda.Stream.Handle);
            output.MarkDeviceModified();
            return result;
        }

        // ---------------------------------------------------------------- routed experts

        /// <summary>Most rows the per-row MoE kernels take; a longer chunk groups its rows by expert (each expert's
        /// weights read once for all its rows), as the per-op prefill does.</summary>
        private const int PerRowExpertRows = 32;

        /// <summary>
        /// A routed-experts region (<see cref="GraphFusion.RoutedExperts"/>) with the per-op path's kernels: the router
        /// (top-K of the logits, softmax over the selected: the renormalized routing) and each row's experts through
        /// device tables of the members' resident copies, all on the device for decode and verify rows; a prompt chunk
        /// routes on the host and runs each selected expert once over its rows. False (the region's ops run) for the
        /// unnormalized routing, members that are not resident, or a geometry the kernels do not take.
        /// </summary>
        private bool TryRoutedExperts(Tensor[] inputs, IGraphWeight[] weights, long[] ip, Tensor[] results)
        {
            if (ip == null || ip.Length < 2 || ip[1] != 1) return false;
            if (weights[2] is not IGraphExpertWeights gate || weights[3] is not IGraphExpertWeights up
                || weights[4] is not IGraphExpertWeights down) return false;
            Tensor x = inputs[0], logits = inputs[1];
            if (x?.Storage is not CudaStorage || logits?.Storage is not CudaStorage
                || x.ElementType != DType.Float32 || logits.ElementType != DType.Float32) return false;
            int experts = (int)gate.Ne2, k = (int)ip[0], hidden = (int)gate.Ne0, ff = (int)gate.Ne1;
            if (up.Ne0 != hidden || up.Ne1 != ff || up.Ne2 != experts || down.Ne0 != ff || down.Ne1 != hidden
                || down.Ne2 != experts || gate.Format != up.Format
                || !CudaFusedOps.IsMoERouterConfigurationSupported(experts, k)) return false;
            long count = x.ElementCount() / hidden;
            if (count < 1 || count > 65535 || x.ElementCount() != count * hidden || logits.ElementCount() != count * experts)
                return false;
            int rows = (int)count;
            var cuda = (CudaAllocator)Allocator;
            RequireDevice(x, cuda.DeviceId, "input", 0);
            RequireDevice(logits, cuda.DeviceId, "input", 1);
            using Tensor x2 = x.View(rows, hidden);
            using Tensor logits2 = logits.View(rows, experts);
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            bool ok = rows <= PerRowExpertRows
                ? PerRowExperts(x2, logits2, output, gate, up, down, experts, k, hidden, ff, rows)
                : GroupedExperts(x2, logits2, output, gate, up, down, experts, k);
            if (!ok)
            {
                output.Dispose();
                return false;
            }
            results[0] = output;
            return true;
        }

        private bool PerRowExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k, int hidden, int ff, int rows)
        {
            Tensor gateTable = ExpertTable(gate), upTable = ExpertTable(up), downTable = ExpertTable(down);
            if (gateTable == null || upTable == null || downTable == null) return false;
            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gate.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(gate.Format, hidden);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(down.Format) && CudaFusedOps.IsMoeDp4aDimensionSupported(down.Format, ff);
            long routed = (long)rows * k;
            using var selected = new Tensor(Allocator, DType.Int32, routed);
            using var routing = new Tensor(Allocator, DType.Float32, routed);
            using var gateOut = new Tensor(Allocator, DType.Float32, routed, ff);
            using var upOut = new Tensor(Allocator, DType.Float32, routed, ff);
            using Tensor inputQ8 = gateUpDp4a ? new Tensor(Allocator, DType.UInt8, (long)rows * (hidden / 32) * CudaFusedOps.Q81BlockBytes) : null;
            using Tensor gateQ8 = downDp4a ? new Tensor(Allocator, DType.UInt8, routed * (ff / 32) * CudaFusedOps.Q81BlockBytes) : null;
            return rows == 1
                ? CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(logits, x, output, selected, routing, gateOut, upOut, IntPtr.Zero,
                    Ptr(gateTable), Ptr(upTable), Ptr(downTable), gate.Format, down.Format, experts, k, hidden, ff,
                    null, IntPtr.Zero, inputQ8, gateQ8, gateUpDp4a, downDp4a)
                : CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(logits, x, output, selected, routing, gateOut, upOut, IntPtr.Zero,
                    Ptr(gateTable), Ptr(upTable), Ptr(downTable), gate.Format, down.Format, experts, k, hidden, ff, rows,
                    null, IntPtr.Zero, inputQ8, gateQ8, gateUpDp4a, downDp4a);
        }

        /// <summary>A prompt chunk's experts: the routing on the host (the device router's selection and weights), then
        /// each selected expert once over its rows (gathered), scattered back weighted.</summary>
        private bool GroupedExperts(Tensor x, Tensor logits, Tensor output, IGraphExpertWeights gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int experts, int k)
        {
            int rows = (int)x.Sizes[0];
            float[] scores = logits.GetElementsAsFloat(rows * experts);
            var byExpert = new List<(int Row, float Weight)>[experts];
            var top = new float[k];
            var index = new int[k];
            for (int r = 0; r < rows; r++)
            {
                RouteRow(scores, r * experts, experts, k, top, index);
                for (int j = 0; j < k; j++)
                    if (index[j] >= 0) (byExpert[index[j]] ??= new()).Add((r, top[j]));
            }
            Ops.Fill(output, 0f);
            for (int e = 0; e < experts; e++)
            {
                var batch = byExpert[e];
                if (batch == null) continue;
                var rowIds = new int[batch.Count];
                var rowWeights = new float[batch.Count];
                for (int i = 0; i < rowIds.Length; i++) (rowIds[i], rowWeights[i]) = batch[i];
                using var ids = new Tensor(Allocator, DType.Int32, rowIds.Length);
                ids.SetElementsAsInt(rowIds);
                using var w = new Tensor(Allocator, DType.Float32, rowWeights.Length);
                w.SetElementsAsFloat(rowWeights);
                using Tensor xs = Ops.IndexSelect(null, x, ids);
                using Tensor g = ExpertMatMul(gate, e, xs);
                using Tensor u = ExpertMatMul(up, e, xs);
                Ops.SiLUMul(g, g, u);
                using Tensor d = ExpertMatMul(down, e, g);
                if (!CudaFusedOps.TryMoEScatterAddWeightedRows(output, d, ids, w)) return false;
            }
            return true;
        }

        /// <summary>ts_moe_router_f32 on the host: the K largest logits (a later equal logit does not displace an
        /// earlier one), softmax over them.</summary>
        private static void RouteRow(float[] logits, int offset, int experts, int k, float[] top, int[] index)
        {
            Array.Fill(top, float.MinValue);
            Array.Fill(index, -1);
            for (int e = 0; e < experts; e++)
            {
                float v = logits[offset + e];
                int min = 0;
                for (int j = 1; j < k; j++) if (top[j] < top[min]) min = j;
                if (v > top[min]) { top[min] = v; index[min] = e; }
            }
            float max = float.MinValue;
            for (int j = 0; j < k; j++) max = MathF.Max(max, top[j]);
            float sum = 0f;
            for (int j = 0; j < k; j++) { top[j] = MathF.Exp(top[j] - max); sum += top[j]; }
            float inv = sum > 0f ? 1f / sum : 0f;
            for (int j = 0; j < k; j++) top[j] *= inv;
        }

        private readonly Dictionary<IGraphExpertWeights, Tensor> _expertTables = new(ReferenceEqualityComparer.Instance);

        /// <summary>A stacked expert weight's table of its members' resident device copies ([experts] pointers, as
        /// int pairs), or null when a member is not resident on this device (then reported once).</summary>
        private Tensor ExpertTable(IGraphExpertWeights set)
        {
            lock (_expertTables)
            {
                if (_expertTables.TryGetValue(set, out Tensor table)) return table;
                var cuda = (CudaAllocator)Allocator;
                var words = new int[2 * set.Ne2];
                for (int e = 0; e < set.Ne2; e++)
                {
                    IGraphWeight member;
                    try { member = set.Expert(e); }
                    catch (NotSupportedException) { member = null; }
                    if (member == null || !CudaQuantizedOps.TryGetResidentDevicePtr(cuda, member.CacheKey, out IntPtr ptr, out int type)
                        || type != set.Format)
                    {
                        Console.Error.WriteLine($"WARNING: CUDA routed experts: expert {e} of a stacked weight is not resident on " +
                            $"device {cuda.DeviceId}; its layer's experts run as separate products.");
                        return _expertTables[set] = null;
                    }
                    long v = ptr.ToInt64();
                    words[2 * e] = unchecked((int)v);
                    words[2 * e + 1] = unchecked((int)(v >> 32));
                }
                table = new Tensor(Allocator, DType.Int32, words.Length);
                table.SetElementsAsInt(words);
                ((CudaStorage)table.Storage).EnsureDeviceCurrent();
                return _expertTables[set] = table;
            }
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
