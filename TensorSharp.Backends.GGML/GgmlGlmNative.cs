// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native GLM (glm-dsa) whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp). The native side loads the
// (split) GGUF itself, places the layers across every visible GPU, owns the
// MLA and lightning-indexer KV caches, and runs prefill / decode ubatches
// through ggml_backend_sched.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlGlmNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        /// <summary>No routed-expert CPU offload (the default).</summary>
        public const int CpuMoeNone = 0;
        /// <summary>Offload only as many leading layers as the VRAM actually needs.</summary>
        public const int CpuMoeAuto = -1;

        static GgmlGlmNative()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        // Paths cross as UTF-8: ggml_fopen decodes them as UTF-8 on Windows,
        // while CharSet.Ansi would marshal the active code page.
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_GlmLoadModel([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int tp, int ctxIsHardLimit, int loadMtp);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmVocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmCtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmNPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmForward(IntPtr handle, int* tokens, int nTokens, float* logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmReset(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmRewind(IntPtr handle, int nPast);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmForwardBatchedDecode(IntPtr handle, int n, int[] slotIds,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmFree(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmQueueVisionRows(IntPtr handle, float* rows, int nRows, int index);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmClearVisionRows(IntPtr handle);

        // ---- NextN/MTP speculative decoding -------------------------------

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHasMtp(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHiddenSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmSpecForward(IntPtr handle, int* tokens, int nTokens,
            float* hOut, float* logitsOut, int allLogitsRows);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmMtpDraftStep(IntPtr handle, int token, float* hPrev, int pos,
            float* logitsOut, float* hOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmMtpCatchUp(IntPtr handle, int* tokens, int nTokens,
            float* hRows, int startPos);

        // ---- glm5next KDA recurrent-state snapshot (speculative rollback) ----

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmKdaStateApiVersion();

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmKdaStateCapture(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmKdaStateRestore(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmResetChecked(IntPtr handle);

        /// <param name="nGpu">GPUs to spread the layers over; 0 = every visible device.</param>
        /// <param name="nCpuMoe">Leading layers whose routed experts stay in system RAM;
        /// <see cref="CpuMoeAuto"/> offloads the fewest that make the model fit.</param>
        /// <param name="backendName">ggml backend registry name ("CUDA" / "Vulkan" / "Metal");
        /// null or empty accepts any GPU backend.</param>
        /// <param name="tp">Tensor-parallel ranks. 1 (the default) layer-splits the
        /// model across the visible GPUs; &gt;1 runs EVERY layer on every rank with
        /// the attention heads and the routed experts sharded, so all the GPUs work
        /// on each token instead of one at a time.</param>
        /// <param name="ctxIsHardLimit">True when <paramref name="nCtx"/> came from
        /// the caller (MAX_CONTEXT) and must be honoured or refused; false when it
        /// is only what the GGUF advertises, which the loader may cap to whatever
        /// the devices can actually hold beside the weights.</param>
        /// <param name="loadMtp">Load the trailing NextN/MTP draft block. It is a
        /// whole extra decoder layer (~3 GiB of GLM-5.2 at IQ2_XXS), so it stays
        /// unloaded unless speculation was actually requested.</param>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads,
            int nCpuMoe = CpuMoeNone, string backendName = null, int tp = 1, bool ctxIsHardLimit = false,
            bool loadMtp = false)
            => TSGgml_GlmLoadModel(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty, tp,
                                   ctxIsHardLimit ? 1 : 0, loadMtp ? 1 : 0);

        public static int VocabSize(IntPtr handle) => TSGgml_GlmVocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_GlmCtxSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_GlmNPast(handle);

        /// <summary>Evaluate <paramref name="tokens"/> at the active slot's current
        /// position. <paramref name="logitsOut"/>, when non-null, receives the last
        /// token's logits.</summary>
        public static unsafe bool Forward(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmForward(handle, t, tokens.Length, l) != 0;
            }
        }

        public static void Reset(IntPtr handle) => TSGgml_GlmReset(handle);

        /// <summary>Queue projected vision-embedding rows (glm5next) to override the
        /// token embeddings of image-placeholder positions in the NEXT Forward call.
        /// <paramref name="index"/> is the first placeholder's position within that
        /// call's token array. The queue is consumed by the forward.</summary>
        public static unsafe bool QueueVisionRows(IntPtr handle, float[] rows, int nRows, int index)
        {
            fixed (float* r = rows)
            {
                return TSGgml_GlmQueueVisionRows(handle, r, nRows, index) != 0;
            }
        }

        /// <summary>Drop queued vision rows (a cancelled or re-planned prompt).</summary>
        public static void ClearVisionRows(IntPtr handle) => TSGgml_GlmClearVisionRows(handle);

        /// <summary>Drop the KV of the tokens after <paramref name="nPast"/>.</summary>
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_GlmRewind(handle, nPast) != 0;

        /// <summary>One decode step for several sequences at once — one token per
        /// sequence, sharing a single read of the weights. Returns false when the
        /// native side declines the batch (tensor parallelism, a position that
        /// disagrees with its slot, a repeated slot), in which case the caller
        /// stays on the per-sequence path.</summary>
        public static bool ForwardBatchedDecode(IntPtr handle, int[] slotIds, int[] tokens,
                                                int[] positions, float[] logitsOut)
            => TSGgml_GlmForwardBatchedDecode(handle, slotIds.Length, slotIds, tokens, positions, logitsOut) != 0;

        /// <summary>Allocate a sequence slot (own KV caches, shared weights);
        /// -1 on failure.</summary>
        public static int SlotAlloc(IntPtr handle) => TSGgml_GlmSlotAlloc(handle);

        /// <summary>Select the slot Forward / Reset / NPast act on.</summary>
        public static bool SetActiveSlot(IntPtr handle, int slotId) => TSGgml_GlmSetActiveSlot(handle, slotId) != 0;

        /// <summary>Free a slot's caches and the graphs built against them.</summary>
        public static bool SlotFree(IntPtr handle, int slotId) => TSGgml_GlmSlotFree(handle, slotId) != 0;

        public static void Free(IntPtr handle) => TSGgml_GlmFree(handle);

        // ---- NextN/MTP speculative decoding -------------------------------

        /// <summary>True when the trailing NextN/MTP block was loaded and is usable.</summary>
        public static bool HasDraftHead(IntPtr handle) => TSGgml_GlmHasMtp(handle) != 0;

        /// <summary>Hidden size (n_embd) — the width of one h_nextn row.</summary>
        public static int HiddenSize(IntPtr handle) => TSGgml_GlmHiddenSize(handle);

        /// <summary>Trunk forward that also captures the post-output_norm hidden
        /// state of every row into <paramref name="hOut"/>. With
        /// <paramref name="allLogitsRows"/> the LM head runs on every row
        /// (n*vocab floats out); otherwise only the last row's logits are written.</summary>
        public static unsafe bool SpecForward(IntPtr handle, int[] tokens, float[] hOut,
                                              float[] logitsOut, bool allLogitsRows)
        {
            fixed (int* t = tokens)
            fixed (float* h = hOut)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmSpecForward(handle, t, tokens.Length, h, l, allLogitsRows ? 1 : 0) != 0;
            }
        }

        /// <summary>One NextN draft step at <paramref name="pos"/>.</summary>
        public static unsafe bool DraftStep(IntPtr handle, int token, float[] hPrev, int pos,
                                               float[] logitsOut, float[] hOut)
        {
            fixed (float* hp = hPrev)
            fixed (float* l = logitsOut)
            fixed (float* h = hOut)
            {
                return TSGgml_GlmMtpDraftStep(handle, token, hp, pos, l, h) != 0;
            }
        }

        /// <summary>Replay verified tokens through the NextN block so its KV cache
        /// tracks the trunk. Row k of <paramref name="hRows"/> is the trunk hidden
        /// state of the token preceding <c>tokens[k]</c>.</summary>
        public static unsafe bool DraftCatchUp(IntPtr handle, int[] tokens, float[] hRows, int startPos)
        {
            fixed (int* t = tokens)
            fixed (float* h = hRows)
            {
                return TSGgml_GlmMtpCatchUp(handle, t, tokens.Length, h, startPos) != 0;
            }
        }

        // ---- glm5next KDA recurrent-state snapshot (speculative rollback) ----

        /// <summary>True when the loaded native library exports the KDA snapshot
        /// API. A library that predates it still loads and runs glm5next, but
        /// cannot undo a partially rejected speculative window, so the model
        /// declines speculation on it instead of failing mid-verify.</summary>
        public static bool KdaStateApiAvailable()
        {
            try { return TSGgml_GlmKdaStateApiVersion() >= 1; }
            catch (EntryPointNotFoundException) { return false; }
            catch (DllNotFoundException) { return false; }
        }

        /// <summary>Copy the active slot's KDA recurrent state (every rank and
        /// layer, device-to-device) into the executor's snapshot arena and record
        /// the slot's position. Taken right before a speculative verify batch.</summary>
        public static bool KdaStateCapture(IntPtr handle) => TSGgml_GlmKdaStateCapture(handle) != 0;

        /// <summary>Copy the snapshot back into the active slot and rewind it to the
        /// captured position. Returns that position, or -1 when no usable snapshot
        /// exists for the active slot (never taken, taken of another slot, or
        /// invalidated by a reset / free).</summary>
        public static int KdaStateRestore(IntPtr handle) => TSGgml_GlmKdaStateRestore(handle);

        public static bool ResetChecked(IntPtr handle)
        {
            try { return TSGgml_GlmResetChecked(handle) != 0; }
            catch (EntryPointNotFoundException)
            {
                // Older libraries cannot arm KDA speculation. Preserve their
                // existing ordinary reset contract without requiring a rebuild.
                Reset(handle);
                return true;
            }
        }
    }
}
