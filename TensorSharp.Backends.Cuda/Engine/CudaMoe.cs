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
    /// <summary>
    /// One device's expert staging (expert residency): experts a call selects that have no VRAM slot are
    /// copied into <see cref="Staging"/> (at most <see cref="Capacity"/> per call) and their table entries
    /// point there for the call. Device buffers: StageOf [nExpert] (-1 between calls), StageList
    /// [Capacity], StageCount [1], and StageFrom [1] (the count before a call's plan, when calls keep what earlier
    /// ones staged: <see cref="CudaMoe.Experts"/> keepStaged).
    /// </summary>
    internal sealed class CudaExpertStage
    {
        public IntPtr StageOf, StageList, StageCount, StageFrom, Staging;
        public long Stride;
        public int Capacity;
    }

    /// <summary>One layer's view for staging: its tables, which experts live in host memory, tables of each
    /// expert's device-mapped host copy (what staging copies from) and of the address each table entry
    /// returns to after a call (the host copy, or the zero expert when the host computes missed experts),
    /// per-expert bytes, and where up/down sit in a staging slot.</summary>
    internal struct CudaExpertStageLayer
    {
        public IntPtr IsHost, TabGate, TabUp, TabDown, SrcTabGate, SrcTabUp, SrcTabDown, RestTabGate, RestTabUp, RestTabDown;
        public long BytesGate, BytesUp, BytesDown, OffUp, OffDown;
        /// <summary>Whether calls stage this layer's missed experts (expert sizes in whole 16-byte units).</summary>
        public bool Staged;
        public bool IsValid => IsHost != IntPtr.Zero;
    }

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
        private const int TQ5_1 = 7, TQ8_0 = 8, TQ2_K = 10, TQ3_K = 11, TQ4_K = 12, TQ5_K = 13, TQ6_K = 14,
            TIQ2_XXS = 16, TIQ2_XS = 17, TIQ3_XXS = 18, TIQ1_S = 19, TIQ4_NL = 20, TIQ3_S = 21, TIQ2_S = 22,
            TIQ4_XS = 23, TMXFP4 = 39, TQ2_0 = 42;

        /// <summary>Every expert layout the kernels decode, as a refusal lists them.</summary>
        public const string SupportedExpertTypes =
            "Q8_0, Q6_K, Q5_K, Q5_1, Q4_K, Q3_K, Q2_K, Q2_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, MXFP4";

        /// <summary>The layouts whose decoders came with the module's low-bit IQ marker
        /// (<see cref="Dsv4Kernels.DecodesLowBitIqExperts"/>): IQ1_S and IQ2_S on every route, IQ2_XXS
        /// on the grouped ones. A module without it reads those rows as Q8_0. IQ2_XXS's warp-per-row
        /// decoder (the DSpark drafter's) is older, so there IQ2_XXS still runs on those kernels alone.
        /// Unsloth's DeepSeek V4.1 1017 UD-IQ2_XXS keeps its gate/up experts in these three.</summary>
        private static bool LowBitIqType(int type) => type == TIQ1_S || type == TIQ2_S || type == TIQ2_XXS;

        /// <summary>The layouts the grouped expert kernels (tensor-core and register-staged) decode
        /// on a module with (<paramref name="decodesLowBitIq"/>) or without the low-bit IQ marker; the
        /// default answers for a module without it, which keeps IQ2_XXS on the warp-per-row kernels.
        /// IQ2_XS and IQ3_XXS are Qwen3.8-Flash-Next UD-Q2_K_XL's gate/up experts: on the warp-per-row
        /// fallback its 1,823-token prefill ran at 500 tok/s on two A40s, against ggml-cuda's
        /// 1,174-1,217 and 1,612 once these kernels took the two layouts. Q5_1 is its UD-Q4_K_XL's
        /// down projection in 43 of 48 layers.</summary>
        public static bool GroupedSupportsType(int type, bool decodesLowBitIq = false, bool decodesLegacy = false)
            => type == TIQ3_S || type == TMXFP4 || type == TQ8_0 || type == TQ6_K || type == TQ5_K || type == TQ4_K
                || type == TQ3_K || type == TQ2_K || type == TIQ4_NL || type == TIQ4_XS
                || type == TIQ2_XS || type == TIQ3_XXS || type == TQ5_1
                || (decodesLowBitIq && LowBitIqType(type))
                || (decodesLegacy && type == TQ2_0);

        /// <summary>Whether some expert route of a module built from the current kernel source decodes
        /// this layout (<see cref="SupportedExpertTypes"/>); <see cref="RequireExpertDecoders"/> answers
        /// for the module actually loaded.</summary>
        public static bool SupportsExpertType(int type) => GroupedSupportsType(type, decodesLowBitIq: true, decodesLegacy: true);

        /// <summary>The kernels <see cref="Experts"/> runs a ubatch on.</summary>
        internal enum ExpertRoute { PerSlot, TensorCore, Staged, Warp }

        /// <summary>
        /// TS_Q2_0_EXPERTS_INT8: whether the tensor-core route runs Q2_0 experts (ISTA's Qwen3.8-Flash-Next pack) on
        /// the int8 kernels (ts_dsv4_moe_q2i8_*: the codes as int8 operands against q8_1-style activations with F32
        /// scales, SwiGLU and the down product's activations fused) instead of widening the weights to F16. On by
        /// default; <c>0</c> keeps the F16 kernel. The activations round as the decode kernels' (and ggml-cuda's MMQ)
        /// q8_1 rows do, so a prompt's expert outputs carry the decode steps' error instead of the F16 kernel's.
        /// Settable for in-process A/B runs (a real-model test compares both routes on one loaded model).
        /// </summary>
        internal static bool Q2_0ExpertsInt8 { get; set; } = Environment.GetEnvironmentVariable("TS_Q2_0_EXPERTS_INT8") != "0";

        /// <summary>Test hook for real-model A/B runs: every ubatch takes the per-slot decode kernels (q8_1 activations
        /// against the integer codes, the arithmetic of ggml-cuda's MMQ/MMVQ), so one prompt's experts can be compared in
        /// all three arithmetics on one loaded model. Never set outside tests.</summary>
        internal static bool PerSlotForTest { get; set; }

        /// <summary>
        /// Whether a tensor-core ubatch takes the int8 Q2_0 kernels: switched on (<paramref name="enabled"/>), a module
        /// with their bodies, Q2_0 gate/up and down, widths on their 128 grid, the tokens' int8 activations fitting in
        /// the scratch's up outputs (where they are written; H goes to the gate outputs), and every stacked weight on a
        /// 4-byte boundary (their rows are copied in 4-byte words; residency tables point at aligned slots).
        /// </summary>
        internal static bool UsesQ2I8(bool enabled, bool hasKernels, in DeviceWeight gate, in DeviceWeight up,
            in DeviceWeight down, int e, int ff, int nUsed)
        {
            static bool Aligned(in DeviceWeight w) => w.Table != IntPtr.Zero || ((long)w.Ptr & 3) == 0;
            return enabled && hasKernels && gate.Type == TQ2_0 && up.Type == TQ2_0 && down.Type == TQ2_0
                && Dsv4Kernels.MoeQ2I8Supports(e, ff)
                && Dsv4Kernels.MoeQ2I8ActRowsBytes(1, e) <= (long)nUsed * ff * sizeof(float)
                && Aligned(gate) && Aligned(up) && Aligned(down);
        }

        /// <summary>
        /// Up to <paramref name="perSlotMaxRows"/> tokens run the per-(token, expert) decode kernels;
        /// longer ubatches the tensor-core kernel where the module has its body and both widths sit on
        /// its k-step, else the register-staged kernels where they take both widths, else the grouped
        /// warp-per-row kernels. Both grouped routes need gate/up and down in layouts
        /// <see cref="GroupedSupportsType"/> grants this module.
        /// </summary>
        internal static ExpertRoute SelectRoute(int nt, int perSlotMaxRows, bool hasMoeMma, bool decodesLowBitIq,
            int guType, int downType, int e, int ff, bool decodesLegacy = false)
        {
            if (nt <= perSlotMaxRows)
                return ExpertRoute.PerSlot;
            bool grouped = GroupedSupportsType(guType, decodesLowBitIq, decodesLegacy)
                && GroupedSupportsType(downType, decodesLowBitIq, decodesLegacy);
            // hasMoeMma: from the committed compute_75 PTX the tensor-core kernel compiles to an
            // empty body and would leave ffnOut unwritten; the staged route serves those builds.
            if (grouped && hasMoeMma && e % Dsv4Kernels.MoeMmaK == 0 && ff % Dsv4Kernels.MoeMmaK == 0)
                return ExpertRoute.TensorCore;
            if (grouped && Dsv4Kernels.StagedSupports(e) && Dsv4Kernels.StagedSupports(ff))
                return ExpertRoute.Staged;
            return ExpertRoute.Warp;
        }

        /// <summary>
        /// Refuses a layer whose routed experts the kernels cannot run, from the loader's descriptors:
        /// an engine calls it for every layer before it creates a device context or places a byte,
        /// so the refusal reaches the host as the one <see cref="NotSupportedException"/> naming the
        /// tensor, and not from inside the per-device upload, where several devices refusing at once
        /// left the host an AggregateException it could not report.
        /// </summary>
        public static void RequireExpertWeights(string tag, int layer,
            in CudaWeightDesc gate, in CudaWeightDesc up, in CudaWeightDesc down)
        {
            RequireExpertWeight(tag, layer, "gate", gate);
            RequireExpertWeight(tag, layer, "up", up);
            RequireExpertWeight(tag, layer, "down", down);
            if (gate.GgmlType != up.GgmlType)
            {
                throw new NotSupportedException(
                    $"[{tag}] {ExpertName(layer, "gate", gate)} is {GgmlTypeName(gate.GgmlType)} but " +
                    $"{ExpertName(layer, "up", up)} is {GgmlTypeName(up.GgmlType)}; the direct-CUDA expert kernels " +
                    "run gate and up in one launch of one layout. Run this model with --backend ggml_cuda.");
            }
        }

        private static void RequireExpertWeight(string tag, int layer, string role, in CudaWeightDesc w)
        {
            if (!SupportsExpertType(w.GgmlType))
            {
                throw new NotSupportedException(
                    $"[{tag}] {ExpertName(layer, role, w)} is {GgmlTypeName(w.GgmlType)} (ggml type {w.GgmlType}), " +
                    $"which the direct-CUDA expert kernels do not decode (they take {SupportedExpertTypes}). " +
                    "Run this model with --backend ggml_cuda.");
            }
        }

        /// <summary>
        /// Refuses experts whose decoder the loaded kernel module predates (built from older source, such
        /// as a committed PTX baseline not yet regenerated), whose expert kernels would read the rows as
        /// Q8_0 and answer with wrong output and no error: Q5_1 unless <paramref name="decodesQ5_1"/>
        /// (<see cref="Dsv4Kernels.DecodesQ5_1Experts"/>), IQ1_S and IQ2_S unless
        /// <paramref name="decodesLowBitIq"/> (<see cref="Dsv4Kernels.DecodesLowBitIqExperts"/>; the
        /// default refuses them). IQ2_XXS passes either way: <see cref="GroupedSupportsType"/> keeps it
        /// on such a module's warp-per-row kernels. Every device loads the same module file.
        /// </summary>
        public static void RequireExpertDecoders(string tag, bool decodesQ5_1, int layer,
            in CudaWeightDesc gate, in CudaWeightDesc up, in CudaWeightDesc down, bool decodesLowBitIq = false,
            bool decodesLegacy = false)
        {
            if (decodesQ5_1 && decodesLowBitIq && decodesLegacy)
                return;
            foreach ((CudaWeightDesc w, string role) in new[] { (gate, "gate"), (up, "up"), (down, "down") })
            {
                bool decoded = w.GgmlType switch
                {
                    TQ5_1 => decodesQ5_1,
                    TIQ1_S or TIQ2_S => decodesLowBitIq,
                    TQ2_0 => decodesLegacy,
                    _ => true,
                };
                if (!decoded)
                {
                    string type = GgmlTypeName(w.GgmlType);
                    throw new NotSupportedException(
                        $"[{tag}] {ExpertName(layer, role, w)} is {type}, and the loaded tensorsharp_dsv4_kernels module " +
                        $"was built from kernel source older than its {type} expert decoder (a committed PTX baseline not " +
                        "yet regenerated). Rebuild TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH, or run this model with " +
                        "--backend ggml_cuda.");
                }
            }
        }

        private static string ExpertName(int layer, string role, in CudaWeightDesc w)
            => string.IsNullOrEmpty(w.Name) ? $"blk.{layer}.ffn_{role}_exps.weight" : w.Name;

        /// <summary>ggml's name for a tensor type id, for messages.</summary>
        internal static string GgmlTypeName(int type) => type switch
        {
            0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
            10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K",
            16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S",
            22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64",
            29 => "IQ1_M", 30 => "BF16", 34 => "TQ1_0", 35 => "TQ2_0", 39 => "MXFP4", 40 => "NVFP4", 42 => "Q2_0",
            _ => $"type {type}",
        };

        /// <summary>
        /// ffnOut[t] = sum_j selW[t, j] * expert_{sel[t, j]}(cur[t]) + shDown[t]. Up to
        /// <paramref name="perSlotMaxRows"/> tokens (a decode step, a batched decode step, a
        /// verify window) run the decode kernels once per (token, selected expert) slot, so every
        /// token computes exactly what it computes alone. Longer ubatches group the slots by
        /// expert and run on tensor cores when the widths allow, else through the
        /// register-staged kernels.
        /// </summary>
        /// <param name="keepStaged">The call leaves its staged experts in place for the next call on the same layer
        /// (which then copies only experts not staged yet); the caller unstages with <see cref="Unstage"/> once the
        /// layer's calls are done. Needs <see cref="Dsv4Kernels.HasStageKeep"/>.</param>
        public static void Experts(Dsv4Kernels dk, CudaKernels kernels, CudaMoeScratch s,
            in DeviceWeight gate, in DeviceWeight up, in DeviceWeight down,
            Tensor cur, Tensor shDown, Tensor ffnOut, int nt, int nUsed, int nExpert, int e, int ff, float clamp,
            int perSlotMaxRows, IntPtr stream, bool? fused = null,
            CudaExpertStage stage = null, in CudaExpertStageLayer stageLayer = default, IntPtr shDownReady = default,
            bool keepStaged = false, IntPtr curQ81 = default, bool? q2i8 = null)
        {
            if (PerSlotForTest)
                perSlotMaxRows = int.MaxValue;
            int guType = gate.Type, downType = down.Type;
            int slots = nt * nUsed;
            bool staging = stage != null && stageLayer.IsValid && stageLayer.Staged;
            bool keep = staging && keepStaged && dk.HasStageKeep && stage.StageFrom != IntPtr.Zero;
            if (keep)
            {
                dk.MoeStageMarkFrom(stage, stream);
                dk.MoeStagePlan(s.Sel, slots, stageLayer, stage, stream);
                dk.MoeStageCopyNew(stageLayer, stage, slots, stream);
            }
            else if (staging)
            {
                dk.MoeStagePlan(s.Sel, slots, stageLayer, stage, stream);
                dk.MoeStageCopy(stageLayer, stage, slots, stream);
            }
            RunExperts(dk, kernels, s, gate, up, down, cur, shDown, ffnOut, nt, nUsed, nExpert, e, ff, clamp,
                perSlotMaxRows, stream, fused, shDownReady, curQ81, q2i8 ?? Q2_0ExpertsInt8);
            if (staging && !keep)
                dk.MoeUnstage(stageLayer, stage, stream);
        }

        /// <summary>Ends a layer's run of <see cref="Experts"/> calls that kept their staging.</summary>
        public static void Unstage(Dsv4Kernels dk, CudaExpertStage stage, in CudaExpertStageLayer stageLayer, IntPtr stream)
        {
            if (stage != null && stageLayer.IsValid && stageLayer.Staged)
                dk.MoeUnstage(stageLayer, stage, stream);
        }

        private static void RunExperts(Dsv4Kernels dk, CudaKernels kernels, CudaMoeScratch s,
            in DeviceWeight gate, in DeviceWeight up, in DeviceWeight down,
            Tensor cur, Tensor shDown, Tensor ffnOut, int nt, int nUsed, int nExpert, int e, int ff, float clamp,
            int perSlotMaxRows, IntPtr stream, bool? fused, IntPtr shDownReady = default, IntPtr curQ81 = default,
            bool q2i8 = false)
        {
            int guType = gate.Type, downType = down.Type;
            int slots = nt * nUsed;
            ExpertRoute route = SelectRoute(nt, perSlotMaxRows, dk.HasMoeMma, dk.DecodesLowBitIqExperts, guType, downType, e, ff,
                dk.DecodesLegacyQuants);
            bool perSlot = route == ExpertRoute.PerSlot;
            bool mma = route == ExpertRoute.TensorCore;
            bool int8 = mma && UsesQ2I8(q2i8, dk.HasMoeQ2I8, gate, up, down, e, ff, nUsed);
            bool staged = route == ExpertRoute.Staged;
            // The per-slot route reads cur's q8_1 rows: the caller's (curQ81, rounded as QuantizeQ81 rounds) when it has them.
            bool given = perSlot && curQ81 != IntPtr.Zero;
            if (staged)
                QuantizeQ81Split(kernels, cur, s.SplitQsA, s.SplitDA, e, nt, stream);
            else if (!mma && !given)
                QuantizeQ81(kernels, cur, s.ActQ8A, e, nt, stream);

            if (perSlot)
            {
                dk.MoeGateUpDecode(gate.Ptr, up.Ptr, given ? curQ81 : Dsv4CudaEngine.Ptr(s.ActQ8A), s.Sel, s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes,
                    nt, nUsed, stream, gate.Table, up.Table);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.ActQ8B, null, ff, slots, clamp, stream, fused);
                dk.MoeDownDecode(down.Ptr, s.ActQ8B, s.Sel, s.ExpDown, downType, e, ff, down.RowBytes, slots, stream, down.Table);
                // A shared expert computed on another stream (a fork in the caller's step) joins here, where its output is read.
                if (shDownReady != IntPtr.Zero)
                    CudaDriverApi.cuStreamWaitEvent(stream, shDownReady, 0).ThrowOnError();
                dk.MoeScatterAdd(s.ExpDown, null, s.SelW, shDown, ffnOut, nt, nUsed, e, stream);
                return;
            }

            // The grouping plan, all on the device.
            kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(s.Counts), nExpert, 0f, stream);
            dk.MoeCount(s.Sel, s.Counts, slots, stream);
            dk.MoeScan(s.Counts, s.Offsets, s.Cursors, nExpert, stream);
            dk.MoeScatter(s.Sel, s.Cursors, s.RowOfSlot, s.SlotToken, slots, nUsed, stream);

            if (int8)
            {
                // The tokens' int8 activations live in the up outputs' memory and H in the gate outputs': this route
                // writes neither as F32 (UsesQ2I8 checked the first fits; H takes 80 of every 256 bytes).
                IntPtr xa = Dsv4CudaEngine.Ptr(s.ExpUp), hq = Dsv4CudaEngine.Ptr(s.ExpGate);
                dk.MoeQ2I8QuantAct(Dsv4CudaEngine.Ptr(cur), xa, nt, e, stream);
                dk.MoeQ2I8GateUp(gate.Ptr, up.Ptr, xa, s.Counts, s.Offsets, s.SlotToken, hq, ff, e, gate.RowBytes, clamp,
                    nExpert, stream, gate.Table, up.Table);
                dk.MoeQ2I8Down(down.Ptr, hq, s.Counts, s.Offsets, s.ExpDown, e, ff, down.RowBytes, nExpert, stream, down.Table);
            }
            else if (mma)
            {
                dk.MoeMma(gate.Ptr, up.Ptr, cur, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream, gate.Table, up.Table);
                SwigluClamp(s.ExpGate, s.ExpUp, (long)slots * ff, clamp);
                dk.MoeMma(down.Ptr, IntPtr.Zero, s.ExpGate, s.Counts, s.Offsets, null,
                    s.ExpDown, null, downType, e, ff, down.RowBytes, nExpert, stream, down.Table);
            }
            else if (staged)
            {
                // Staged-weight kernels decode each expert weight row once into registers and
                // reuse it across the expert's member tokens.
                dk.MoeGateUpStaged(gate.Ptr, up.Ptr, s.SplitQsA, s.SplitDA, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream, gate.Table, up.Table);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.SplitQsB, s.SplitDB, ff, slots, clamp, stream, fused);
                dk.MoeDownStaged(down.Ptr, s.SplitQsB, s.SplitDB, s.Counts, s.Offsets, s.ExpDown,
                    downType, e, ff, down.RowBytes, nExpert, stream, down.Table);
            }
            else
            {
                dk.MoeGateUp(gate.Ptr, up.Ptr, s.ActQ8A, s.Counts, s.Offsets, s.SlotToken,
                    s.ExpGate, s.ExpUp, guType, ff, e, gate.RowBytes, nExpert, stream, gate.Table, up.Table);
                SwigluQuantize(kernels, s.ExpGate, s.ExpUp, s.ActQ8B, null, ff, slots, clamp, stream, fused);
                dk.MoeDown(down.Ptr, s.ActQ8B, s.Counts, s.Offsets, s.ExpDown,
                    downType, e, ff, down.RowBytes, nExpert, stream, down.Table);
            }
            if (shDownReady != IntPtr.Zero)
                CudaDriverApi.cuStreamWaitEvent(stream, shDownReady, 0).ThrowOnError();
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
