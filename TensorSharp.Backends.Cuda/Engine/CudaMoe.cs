// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The routed half of a mixture-of-experts block for the whole-model direct-CUDA engines
// (DeepSeek V4, GLM-5.3-Flash): each token's selected experts and their weights come in,
// their weighted outputs plus the shared expert's go out. The router itself, and the shared
// expert, stay with the model; the expert kernels live in the DeepSeek V4 module.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>One device's routed-expert scratch, sized for a ubatch.</summary>
    internal sealed class CudaMoeScratch
    {
        private const int Q81BlockBytes = 36;

        /// <summary>Each token's selected experts and their weights [nt, nUsed], written by the router.</summary>
        public readonly Tensor Sel, SelW;
        // The grouping plan: per-expert histogram, its exclusive scan and fill cursors, and each
        // (token, selection) slot's row in expert order and the token of each such row.
        public readonly Tensor Counts, Offsets, Cursors, RowOfSlot, SlotToken;
        // Gate/up (A) and down (B) execute sequentially on one stream. Their quantized
        // activations have disjoint lifetimes; interleaved and split are exclusive paths.
        // These fields can therefore refer to the same capacity-sized allocation.
        public readonly Tensor ActQ8A, ActQ8B, SplitQsA, SplitDA, SplitQsB, SplitDB;
        /// <summary>Expert outputs per slot row [nt * nUsed, width].</summary>
        public readonly Tensor ExpGate, ExpUp, ExpDown;

        public CudaMoeScratch(Func<DType, long[], Tensor> alloc, int nUbatch, int nExpert, int nUsed, int e, int ff,
            bool? shareActivations = null)
        {
            long nt = nUbatch, s = (long)nUbatch * nUsed;
            Sel = alloc(DType.Int32, new[] { nt, nUsed });
            SelW = alloc(DType.Float32, new[] { nt, nUsed });
            // Counts is cleared with the F32 fill kernel: 0.0f and 0 share a bit pattern, and
            // that keeps the clear stream-ordered with the grouping kernels.
            Counts = alloc(DType.Int32, new[] { (long)nExpert });
            Offsets = alloc(DType.Int32, new[] { (long)nExpert });
            Cursors = alloc(DType.Int32, new[] { (long)nExpert });
            RowOfSlot = alloc(DType.Int32, new[] { s });
            SlotToken = alloc(DType.Int32, new[] { s });
            if (shareActivations ?? CudaKernels.MoeFusionEnabled)
            {
                long blocks = Math.Max(nt * (e / 32), s * (ff / 32));
                ActQ8A = ActQ8B = SplitQsA = SplitQsB = alloc(DType.UInt8, new[] { blocks * Q81BlockBytes });
                SplitDA = SplitDB = alloc(DType.Float32, new[] { blocks });
            }
            else
            {
                ActQ8A = alloc(DType.UInt8, new[] { nt, (long)(e / 32) * Q81BlockBytes });
                ActQ8B = alloc(DType.UInt8, new[] { s, (long)(ff / 32) * Q81BlockBytes });
                SplitQsA = alloc(DType.UInt8, new[] { nt, (long)e });
                SplitDA = alloc(DType.Float32, new[] { nt, (long)e / 32 });
                SplitQsB = alloc(DType.UInt8, new[] { s, (long)ff });
                SplitDB = alloc(DType.Float32, new[] { s, (long)ff / 32 });
            }
            ExpGate = alloc(DType.Float32, new[] { s, (long)ff });
            ExpUp = alloc(DType.Float32, new[] { s, (long)ff });
            ExpDown = alloc(DType.Float32, new[] { s, (long)e });
        }

        /// <summary>Bytes of scratch a device needs for these shapes, for placement.</summary>
        public static long Bytes(int nUbatch, int nUsed, int e, int ff, int nExpert = 0, bool? shareActivations = null)
        {
            long s = (long)nUbatch * nUsed;
            long aBlocks = (long)nUbatch * (e / 32), bBlocks = s * (ff / 32);
            long activations = (shareActivations ?? CudaKernels.MoeFusionEnabled)
                ? Math.Max(aBlocks, bBlocks) * (Q81BlockBytes + 4)
                : (aBlocks + bBlocks) * (Q81BlockBytes + 32 + 4);
            return 2 * s * ff * 4 + s * e * 4           // ExpGate / ExpUp / ExpDown
                + activations + 16 * s + 12L * nExpert; // activations, router and grouping plan
        }
    }

    internal static class CudaMoe
    {
        // ggml type ids of the expert weights the kernels take.
        private const int TQ8_0 = 8, TQ2_K = 10, TQ3_K = 11, TQ4_K = 12, TQ5_K = 13, TQ6_K = 14,
            TIQ2_XXS = 16, TIQ2_XS = 17, TIQ3_XXS = 18, TIQ4_NL = 20, TIQ3_S = 21, TIQ4_XS = 23, TMXFP4 = 39;

        /// <summary>Layouts only the warp-per-row kernels decode (the per-token decode kernels and
        /// the grouped warp kernels a long ubatch falls back to): IQ2_XXS, the DSpark drafter's.</summary>
        private static bool RowKernelOnlyType(int type) => type == TIQ2_XXS;

        /// <summary>The layouts the grouped expert kernels (tensor-core and register-staged)
        /// decode; see <see cref="RowKernelOnlyType"/> for the rest. IQ2_XS and IQ3_XXS are
        /// Qwen3.8-Flash-Next UD-Q2_K_XL's gate/up experts: on the warp-per-row fallback its
        /// 1,823-token prefill ran at 500 tok/s on two A40s, against ggml-cuda's 1,174-1,217 and
        /// 1,612 once these kernels took the two layouts.</summary>
        public static bool GroupedSupportsType(int type)
            => type == TIQ3_S || type == TMXFP4 || type == TQ8_0 || type == TQ6_K || type == TQ5_K || type == TQ4_K
                || type == TQ3_K || type == TQ2_K || type == TIQ4_NL || type == TIQ4_XS
                || type == TIQ2_XS || type == TIQ3_XXS;

        public static void RequireExpertType(string tag, int type)
        {
            if (!GroupedSupportsType(type) && !RowKernelOnlyType(type))
            {
                throw new NotSupportedException(
                    $"[{tag}] unsupported expert quant type {type} (supported: Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_XS, IQ2_XXS, MXFP4)");
            }
        }

        /// <summary>
        /// ffnOut[t] = sum_j selW[t, j] * expert_{sel[t, j]}(cur[t]) + shDown[t]. Up to
        /// <paramref name="perSlotMaxRows"/> tokens (a decode step, a batched decode step, a
        /// verify window) run the decode kernels once per (token, selected expert) slot, so every
        /// token computes exactly what it computes alone. Longer ubatches group the slots by
        /// expert and run on tensor cores when the widths allow, else through the
        /// register-staged kernels.
        /// </summary>
        public static void Experts(Dsv4Kernels dk, CudaKernels kernels, CudaMoeScratch s,
            in DeviceWeight gate, in DeviceWeight up, in DeviceWeight down,
            Tensor cur, Tensor shDown, Tensor ffnOut, int nt, int nUsed, int nExpert, int e, int ff, float clamp,
            int perSlotMaxRows, IntPtr stream, bool? fused = null)
        {
            int guType = gate.Type, downType = down.Type;
            int slots = nt * nUsed;
            bool perSlot = nt <= perSlotMaxRows;
            bool mma = !perSlot
                && GroupedSupportsType(guType) && GroupedSupportsType(downType)
                && e % Dsv4Kernels.MoeMmaK == 0 && ff % Dsv4Kernels.MoeMmaK == 0;
            bool staged = !perSlot && !mma
                && GroupedSupportsType(guType) && GroupedSupportsType(downType)
                && Dsv4Kernels.StagedSupports(e) && Dsv4Kernels.StagedSupports(ff);
            if (staged)
                QuantizeQ81Split(kernels, cur, s.SplitQsA, s.SplitDA, e, nt, stream);
            else if (!mma)
                QuantizeQ81(kernels, cur, s.ActQ8A, e, nt, stream);

            if (perSlot)
            {
                dk.MoeGateUpDecode(gate.Ptr, up.Ptr, s.ActQ8A, s.Sel, s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes,
                    nt, nUsed, stream);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.ActQ8B, null, ff, slots, clamp, stream, fused);
                dk.MoeDownDecode(down.Ptr, s.ActQ8B, s.Sel, s.ExpDown, downType, e, ff, down.RowBytes, slots, stream);
                dk.MoeScatterAdd(s.ExpDown, null, s.SelW, shDown, ffnOut, nt, nUsed, e, stream);
                return;
            }

            // The grouping plan, all on the device.
            kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(s.Counts), nExpert, 0f, stream);
            dk.MoeCount(s.Sel, s.Counts, slots, stream);
            dk.MoeScan(s.Counts, s.Offsets, s.Cursors, nExpert, stream);
            dk.MoeScatter(s.Sel, s.Cursors, s.RowOfSlot, s.SlotToken, slots, nUsed, stream);

            if (mma)
            {
                dk.MoeMma(gate.Ptr, up.Ptr, cur, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream);
                SwigluClamp(s.ExpGate, s.ExpUp, (long)slots * ff, clamp);
                dk.MoeMma(down.Ptr, IntPtr.Zero, s.ExpGate, s.Counts, s.Offsets, null,
                    s.ExpDown, null, downType, e, ff, down.RowBytes, nExpert, stream);
            }
            else if (staged)
            {
                // Staged-weight kernels decode each expert weight row once into registers and
                // reuse it across the expert's member tokens.
                dk.MoeGateUpStaged(gate.Ptr, up.Ptr, s.SplitQsA, s.SplitDA, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.SplitQsB, s.SplitDB, ff, slots, clamp, stream, fused);
                dk.MoeDownStaged(down.Ptr, s.SplitQsB, s.SplitDB, s.Counts, s.Offsets, s.ExpDown,
                    downType, e, ff, down.RowBytes, nExpert, stream);
            }
            else
            {
                dk.MoeGateUp(gate.Ptr, up.Ptr, s.ActQ8A, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.ActQ8B, null, ff, slots, clamp, stream, fused);
                dk.MoeDown(down.Ptr, s.ActQ8B, s.Counts, s.Offsets, s.ExpDown,
                    downType, e, ff, down.RowBytes, nExpert, stream);
            }
            dk.MoeScatterAdd(s.ExpDown, s.RowOfSlot, s.SelW, shDown, ffnOut, nt, nUsed, e, stream);
        }

        /// <summary>Clamped SwiGLU over the leading <paramref name="n"/> elements, through the
        /// shared Ops.SiLUMulClamp (gate is overwritten in place).</summary>
        public static void SwigluClamp(Tensor gate, Tensor up, long n, float limit)
        {
            using Tensor g = Flat(gate, n);
            using Tensor u = Flat(up, n);
            Ops.SiLUMulClamp(g, g, u, limit);
        }

        internal static void SwigluQuantize(CudaKernels kernels, Tensor gate, Tensor up, Tensor output, Tensor scales,
            int inDim, int rows, float limit, IntPtr stream, bool? fused = null)
            => kernels.LaunchSiluMulClampQuantizeQ81(Dsv4CudaEngine.Ptr(gate), Dsv4CudaEngine.Ptr(up),
                Dsv4CudaEngine.Ptr(output), scales == null ? IntPtr.Zero : Dsv4CudaEngine.Ptr(scales),
                inDim, rows, limit, stream, fused);

        private static Tensor Flat(Tensor t, long n)
        {
            using Tensor flat = t.View(t.ElementCount());
            return flat.Narrow(0, 0, n);
        }

        private static void QuantizeQ81(CudaKernels kernels, Tensor input, Tensor scratch, int inDim, int rows, IntPtr stream)
            => kernels.LaunchQuantizeQ81Rows(Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(scratch), inDim, rows, stream,
                warpCooperative: true);

        /// <summary>Dense split q8_1 (contiguous int8 row + separate per-block scale). Values are
        /// bit-identical to the interleaved layout; the dense form is what makes the staged expert
        /// kernels' activation reads vectorizable.</summary>
        private static void QuantizeQ81Split(CudaKernels kernels, Tensor input, Tensor qs, Tensor d, int inDim, int rows, IntPtr stream)
            => kernels.LaunchQuantizeQ81SplitRows(Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(qs), Dsv4CudaEngine.Ptr(d),
                inDim, rows, stream);
    }
}
