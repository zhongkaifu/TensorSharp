// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial void TSGgml_Qwen4ExpArenaFlushHostPointer(IntPtr pointer);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int TSGgml_Qwen4ExpArenaFlushHostPointerStatus(IntPtr pointer);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int TSGgml_Qwen4ExpArenaDecodeBatched(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr head, IntPtr ple,
            IntPtr kinds, int numLayers, int pleLayer, int numSequences,
            [In] int[] tokens, [In] int[] positions, [In] int[] ropePositions, [In] int[] capacities,
            [In] IntPtr[] kCaches, [In] IntPtr[] vCaches,
            [In] IntPtr[] convStates, [In] IntPtr[] ssmStates, [In] IntPtr[] pleStates,
            [In] IntPtr[] attnDescriptors, [In] IntPtr[] gdnDescriptors, [In] IntPtr[] pleDescriptors,
            [In] IntPtr[] qsaDescriptors, [In] IntPtr[] qsaPositions, [In] int[] qsaPositionCounts,
            int hiddenSize, int hc, int hcLowRank,
            int headDim, int numHeads, int numKvHeads, int rotaryDims,
            float ropeBase, float ropeFrequencyScale, float attentionScale,
            int headKDim, int headVDim, int numKHeads, int numVHeads, int convKernel,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            float eps, int kvCacheType,
            IntPtr tokenEmbedding, int tokenEmbeddingType,
            long tokenEmbeddingDim, long tokenEmbeddingVocab, long tokenEmbeddingBytes,
            IntPtr embeddingRows, IntPtr pleEmbedding,
            IntPtr logits, IntPtr sampled, int wantLogits, int device);
    }

    public partial class GgmlBasicOps
    {
        public static void Qwen4ExpArenaFlushHostPointer(IntPtr pointer)
            => GgmlNative.TSGgml_Qwen4ExpArenaFlushHostPointer(pointer);

        public static int Qwen4ExpArenaFlushHostPointerStatus(IntPtr pointer)
            => GgmlNative.TSGgml_Qwen4ExpArenaFlushHostPointerStatus(pointer);

        /// <summary>Returns 1 on success, 0 before execution declined, and -1
        /// when execution may have advanced recurrent state. A negative result
        /// must fail the requests instead of falling back to solo decode.</summary>
        public static int Qwen4ExpArenaDecodeBatchedStatus(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr head, IntPtr ple,
            IntPtr kinds, int numLayers, int pleLayer, int numSequences,
            int[] tokens, int[] positions, int[] ropePositions, int[] capacities,
            IntPtr[] kCaches, IntPtr[] vCaches, IntPtr[] convStates, IntPtr[] ssmStates, IntPtr[] pleStates,
            IntPtr[] attnDescriptors, IntPtr[] gdnDescriptors, IntPtr[] pleDescriptors,
            IntPtr[] qsaDescriptors, IntPtr[] qsaPositions, int[] qsaPositionCounts,
            int hiddenSize, int hc, int hcLowRank,
            int headDim, int numHeads, int numKvHeads, int rotaryDims,
            float ropeBase, float ropeFrequencyScale, float attentionScale,
            int headKDim, int headVDim, int numKHeads, int numVHeads, int convKernel,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            float eps, int kvCacheType,
            IntPtr tokenEmbedding, int tokenEmbeddingType,
            long tokenEmbeddingDim, long tokenEmbeddingVocab, long tokenEmbeddingBytes,
            IntPtr embeddingRows, IntPtr pleEmbedding, IntPtr logits, IntPtr sampled, bool wantLogits, int device)
            => GgmlNative.TSGgml_Qwen4ExpArenaDecodeBatched(
                ffn, gdn, attn, head, ple, kinds, numLayers, pleLayer, numSequences,
                tokens, positions, ropePositions, capacities, kCaches, vCaches,
                convStates, ssmStates, pleStates, attnDescriptors, gdnDescriptors, pleDescriptors,
                qsaDescriptors, qsaPositions, qsaPositionCounts,
                hiddenSize, hc, hcLowRank, headDim, numHeads, numKvHeads, rotaryDims,
                ropeBase, ropeFrequencyScale, attentionScale, headKDim, headVDim, numKHeads, numVHeads, convKernel,
                numExperts, numExpertsUsed, expertFf, sharedFf, eps, kvCacheType,
                tokenEmbedding, tokenEmbeddingType, tokenEmbeddingDim, tokenEmbeddingVocab, tokenEmbeddingBytes,
                embeddingRows, pleEmbedding, logits, sampled, wantLogits ? 1 : 0, device);
    }
}
