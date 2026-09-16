// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native DeepSeek V4 (Flash) whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp). The native side loads the
// (split) GGUF itself, places the layers across every visible GPU, owns the
// DSV4 KV caches, and runs prefill/decode ubatches through ggml_backend_sched.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static partial class GgmlDeepSeek4Native
    {
        private const string DllName = "GgmlOps";

        static GgmlDeepSeek4Native()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        // Paths must cross as UTF-8: ggml_fopen decodes them as UTF-8 on Windows,
        // while CharSet.Ansi would marshal the active code page.
        [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_Dsv4LoadModel(string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads,
            int nCpuMoe, string backendName);

        [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_Dsv4LoadModelDspark(string ggufPath,
            int nGpu, int nCtx, int nUbatch,
            int nThreads, string dsparkPath, int nCpuMoe,
            string backendName);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4DsparkBlockSize(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv4ForwardSpec(IntPtr handle, int* tokens, int nTokens, float* logitsOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv4DsparkDraft(IntPtr handle, int anchorToken, int* toksOut, float* confOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4Rewind(IntPtr handle, int nPast);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4Truncate(IntPtr handle, int nPast);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4TruncateAlign(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4VocabSize(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4CtxSize(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4NPast(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv4Forward(IntPtr handle, int* tokens, int nTokens, float* logitsOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Dsv4Reset(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4ResetChecked(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Dsv4Free(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotAlloc(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SetActiveSlot(IntPtr handle, int slotId);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotFree(IntPtr handle, int slotId);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotStatus(IntPtr handle, int slotId,
            out int head, out int checkpoint, out int healthy);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotCanReuse(IntPtr handle, int slotId, int cachedHead, int target);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotCanRetain(IntPtr handle, int slotId,
            int retainedCount, ulong budgetPerDevice);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Dsv4SlotReleaseGraphs(IntPtr handle, int slotId);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_Dsv4ForwardBatchedDecode(
            IntPtr handle, int n, int* slotIds, int* tokens, int* positions, float* logitsOut);

        /// <summary>
        /// Routed-expert CPU offload policy passed to the native loader:
        /// <c>0</c> none (the default — offload is opt-in, and a model that does
        /// not fit is refused at load with the number of layers that would make
        /// it fit), <c>N</c> the first N layers, <see cref="int.MaxValue"/> every
        /// layer, <c>-1</c> auto (the fewest leading layers that make the model
        /// fit the visible VRAM; opt-in only).
        /// </summary>
        public const int CpuMoeAuto = -1;

        /// <summary>No routed-expert offload — what an unspecified policy means.</summary>
        public const int CpuMoeNone = 0;

        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads,
            int nCpuMoe = CpuMoeNone, string backendName = null)
            => TSGgml_Dsv4LoadModel(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty);

        /// <summary>Load with a DSpark drafter GGUF (see the DeepSeek V4 card).
        /// A null/empty path is identical to <see cref="LoadModel"/>.</summary>
        public static IntPtr LoadModelWithDspark(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads,
            string dsparkPath, int nCpuMoe = CpuMoeNone, string backendName = null)
            => TSGgml_Dsv4LoadModelDspark(ggufPath, nGpu, nCtx, nUbatch, nThreads, dsparkPath ?? string.Empty, nCpuMoe,
                backendName ?? string.Empty);

        /// <summary>Tokens the DSpark drafter proposes per block, or 0 when no
        /// drafter is loaded.</summary>
        public static int DsparkBlockSize(IntPtr handle) => TSGgml_Dsv4DsparkBlockSize(handle);

        /// <summary>Trunk forward returning logits for EVERY row (the
        /// speculative verify). Advances the cache like Forward.</summary>
        public static unsafe bool ForwardSpec(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(logitsOut);
            if (handle == IntPtr.Zero || tokens.Length == 0) return false;
            int vocabulary = TSGgml_Dsv4VocabSize(handle);
            if (vocabulary <= 0) return false;
            // The native ABI writes every vocabulary row. Widen before the
            // multiplication so an oversized request cannot wrap the check.
            long required = (long)tokens.Length * vocabulary;
            if (logitsOut.LongLength < required)
                throw new ArgumentException($"The output buffer needs at least {required} logits.", nameof(logitsOut));
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_Dsv4ForwardSpec(handle, t, tokens.Length, l) != 0;
            }
        }

        /// <summary>One DSpark block draft from <paramref name="anchorToken"/>
        /// at the cache's current position. Returns the number of proposals.</summary>
        public static unsafe int DsparkDraft(IntPtr handle, int anchorToken, int[] toksOut, float[] confOut)
        {
            ArgumentNullException.ThrowIfNull(toksOut);
            ArgumentNullException.ThrowIfNull(confOut);
            if (handle == IntPtr.Zero) return 0;
            int block = TSGgml_Dsv4DsparkBlockSize(handle);
            if (block <= 0) return 0;
            if (toksOut.Length < block)
                throw new ArgumentException($"The output buffer needs at least {block} token IDs.", nameof(toksOut));
            if (confOut.Length < block)
                throw new ArgumentException($"The output buffer needs at least {block} confidence scores.", nameof(confOut));
            fixed (int* t = toksOut)
            fixed (float* c = confOut)
            {
                return TSGgml_Dsv4DsparkDraft(handle, anchorToken, t, c);
            }
        }

        /// <summary>Drop the KV of rejected speculative tokens (position rewind only).
        /// V4 only; a conversational rewind of any depth is <see cref="Truncate"/>.</summary>
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_Dsv4Rewind(handle, nPast) != 0;

        /// <summary>
        /// Move the active slot's head back to <paramref name="nPast"/> so the next
        /// forward appends there and the K/V of the first <paramref name="nPast"/>
        /// positions is reused. True when the slot now holds exactly that many
        /// positions and continuing is identical to a fresh prefill of them.
        ///
        /// <para>False means the slot was NOT modified and the caller must reset and
        /// re-prefill. That is a normal outcome, not an error: the raw sliding-window
        /// ring is finite, so a rewind further back than the slot's checkpoint can
        /// reach has no correct answer. Callers must never ignore it - continuing at
        /// the stale head silently answers from the wrong positions.</para>
        /// </summary>
        public static bool Truncate(IntPtr handle, int nPast) => TSGgml_Dsv4Truncate(handle, nPast) != 0;

        /// <summary>The multiple a <see cref="Truncate"/> target must be (so no
        /// compression block straddles the new head), or 0 when this model cannot
        /// truncate at all. Align the reuse length DOWN to it rather than losing the
        /// whole prefix to a refusal over one token.</summary>
        public static int TruncateAlign(IntPtr handle) => TSGgml_Dsv4TruncateAlign(handle);

        public static int VocabSize(IntPtr handle) => TSGgml_Dsv4VocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_Dsv4CtxSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_Dsv4NPast(handle);

        public static unsafe bool Forward(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_Dsv4Forward(handle, t, tokens.Length, l) == 0;
            }
        }

        public static void Reset(IntPtr handle) => TSGgml_Dsv4Reset(handle);
        public static bool ResetChecked(IntPtr handle) => TSGgml_Dsv4ResetChecked(handle) != 0;
        public static void Free(IntPtr handle) => TSGgml_Dsv4Free(handle);

        /// <summary>Allocate a new sequence slot (own KV caches, shared
        /// weights). Returns the slot id, or -1 on failure (e.g. VRAM).</summary>
        public static int SlotAlloc(IntPtr handle) => TSGgml_Dsv4SlotAlloc(handle);

        /// <summary>Inspect an explicitly identified V4.1 slot without selecting it.</summary>
        public static bool SlotStatus(IntPtr handle, int slotId, out int head, out int checkpoint, out bool healthy)
        {
            int result = TSGgml_Dsv4SlotStatus(handle, slotId, out head, out checkpoint, out int ok);
            healthy = result != 0 && ok != 0;
            return result != 0;
        }

        public static bool SlotCanReuse(IntPtr handle, int slotId, int cachedHead, int target)
            => TSGgml_Dsv4SlotCanReuse(handle, slotId, cachedHead, target) != 0;

        public static bool SlotCanRetain(IntPtr handle, int slotId, int retainedCount, ulong budgetPerDevice)
            => TSGgml_Dsv4SlotCanRetain(handle, slotId, retainedCount, budgetPerDevice) != 0;

        public static bool SlotReleaseGraphs(IntPtr handle, int slotId)
            => TSGgml_Dsv4SlotReleaseGraphs(handle, slotId) != 0;

        /// <summary>Select the slot Forward/Reset/NPast act on.</summary>
        public static bool SetActiveSlot(IntPtr handle, int slotId)
            => TSGgml_Dsv4SetActiveSlot(handle, slotId) == 0;

        /// <summary>Free a slot's caches (and its cached graphs). The active
        /// slot cannot be freed.</summary>
        public static bool SlotFree(IntPtr handle, int slotId)
            => TSGgml_Dsv4SlotFree(handle, slotId) == 0;

        /// <summary>True token-batched decode: one token per slot in a single
        /// fused graph. logitsOut receives n rows of vocab-size floats; every
        /// slot advances one position on success. Returns false when the step
        /// can't be batched (caller falls back to per-slot forwards).</summary>
        public static unsafe bool ForwardBatchedDecode(
            IntPtr handle, int[] slotIds, int[] tokens, int[] positions, float[] logitsOut)
        {
            fixed (int* s = slotIds)
            fixed (int* t = tokens)
            fixed (int* p = positions)
            fixed (float* l = logitsOut)
            {
                return TSGgml_Dsv4ForwardBatchedDecode(handle, slotIds.Length, s, t, p, l) == 0;
            }
        }
    }
}
