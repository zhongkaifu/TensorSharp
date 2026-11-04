// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Cuda;

public enum CudaNinferDraftHeadPolicy { FullTargetVocabulary, OriginalQ4G64Shortlist }

/// <summary>Original Q4G64_F16S proposal head, independent of GGML Q4_K; no quantization conversion.</summary>
public sealed class CudaNinferDraftShortlistDesc
{
    public const int Rows = 131072, Columns = 5120, GroupSize = 64;
    public const long CodeBytes = (long)Rows * Columns / 2, ScaleBytes = (long)Rows * Columns / 32;
    public ICudaWeightSource Source { get; }
    public long Bytes => CodeBytes + ScaleBytes;
    public IReadOnlyList<int> TokenIds { get; }
    public CudaNinferDraftShortlistDesc(ICudaWeightSource source, long bytes, IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(source); ArgumentNullException.ThrowIfNull(tokenIds);
        if (bytes != CodeBytes + ScaleBytes || tokenIds.Count != Rows)
            throw new ArgumentException("Original NInfer shortlist encoded geometry differs.");
        var copy = tokenIds.ToArray();
        foreach (int id in copy) if ((uint)id >= 248320) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        // The original loader does not impose uniqueness. Preserve all rows and their exact map.
        Source = source; TokenIds = Array.AsReadOnly(copy);
    }
}

public sealed class CudaNinferDraftLayerDesc
{
    public ReadOnlyMemory<float> InputNorm { get; }
    public ReadOnlyMemory<float> QueryNorm { get; }
    public ReadOnlyMemory<float> KeyNorm { get; }
    public ReadOnlyMemory<float> PostAttentionNorm { get; }
    /// <summary>Original [side][tap][channel], including all four sides/taps; no main-body convolution transpose.</summary>
    public ReadOnlyMemory<float> AttentionConvBase { get; }
    public ReadOnlyMemory<float> MlpConvBase { get; }
    public CudaNinferProjectionDesc QueryKeyValue { get; }
    public CudaNinferProjectionDesc AttentionOutput { get; }
    public CudaNinferProjectionDesc AttentionConvProjection { get; }
    public CudaNinferProjectionDesc GateUp { get; }
    public CudaNinferProjectionDesc Down { get; }
    public CudaNinferProjectionDesc MlpConvProjection { get; }
    public CudaNinferDraftLayerDesc(float[] inputNorm, float[] queryNorm, float[] keyNorm, float[] postAttentionNorm,
        float[] attentionConvBase, float[] mlpConvBase, CudaNinferProjectionDesc qkv, CudaNinferProjectionDesc output,
        CudaNinferProjectionDesc attentionConv, CudaNinferProjectionDesc gateUp, CudaNinferProjectionDesc down,
        CudaNinferProjectionDesc mlpConv)
    {
        InputNorm = CudaNinferDraftDesc.Plain(inputNorm, 5120); QueryNorm = CudaNinferDraftDesc.Plain(queryNorm, 128);
        KeyNorm = CudaNinferDraftDesc.Plain(keyNorm, 128); PostAttentionNorm = CudaNinferDraftDesc.Plain(postAttentionNorm, 5120);
        AttentionConvBase = CudaNinferDraftDesc.Plain(attentionConvBase, 20480); MlpConvBase = CudaNinferDraftDesc.Plain(mlpConvBase, 20480);
        QueryKeyValue = CudaNinferDraftDesc.Projection(qkv, 6144, 5120);
        AttentionOutput = CudaNinferDraftDesc.Projection(output, 5120, 4096);
        AttentionConvProjection = CudaNinferDraftDesc.Projection(attentionConv, 1280, 5120);
        GateUp = CudaNinferDraftDesc.Projection(gateUp, 34816, 5120);
        Down = CudaNinferDraftDesc.Projection(down, 5120, 17408);
        MlpConvProjection = CudaNinferDraftDesc.Projection(mlpConv, 1280, 5120);
    }
}

/// <summary>
/// Original DFlash2 loading contract. Sources remain borrowed until synchronous upload completes.
/// It must not be passed to the GGUF drafter: original A16 FMA, BF16 boundaries/rings, and selector kernels are required.
/// Constructing this description does not attach a runnable drafter.
/// </summary>
public sealed class CudaNinferDraftDesc
{
    public const string NumericalPolicy = "original-dflash2-a16-bf16-plain-norm-concatenated-output-taps-v1";
    public const int RingPositions = 2048, QueryHeads = 32, KeyValueHeads = 8, HeadDimension = 128;
    public const int TopK = 16, SelectorRank = 256, ConvTaps = 2, ConvGroupSize = 16, MaskToken = 248070;
    public static IReadOnlyList<int> TargetOutputLayers { get; } = Array.AsReadOnly(new[] { 5, 19, 33, 47, 61 });
    public int DraftTokens { get; }
    public int BlockSize => DraftTokens + 1;
    public CudaNinferDraftHeadPolicy HeadPolicy { get; }
    public CudaNinferDraftShortlistDesc? Shortlist { get; }
    public CudaNinferProjectionDesc FeatureProjection { get; }
    public ReadOnlyMemory<float> ContextNorm { get; }
    public ReadOnlyMemory<float> FinalNorm { get; }
    public IReadOnlyList<CudaNinferDraftLayerDesc> Layers { get; }
    public CudaNinferProjectionDesc SelectorHidden { get; }
    public CudaNinferProjectionDesc SelectorPredecessor { get; }
    public CudaNinferProjectionDesc SelectorSuccessor { get; }
    public string Provenance { get; }
    public CudaNinferDraftDesc(int draftTokens, CudaNinferDraftHeadPolicy headPolicy, CudaNinferDraftShortlistDesc? shortlist,
        CudaNinferProjectionDesc feature, float[] contextNorm, float[] finalNorm,
        IReadOnlyList<CudaNinferDraftLayerDesc> layers, CudaNinferProjectionDesc selectorHidden,
        CudaNinferProjectionDesc selectorPredecessor, CudaNinferProjectionDesc selectorSuccessor, string provenance)
    {
        if (draftTokens < 1 || draftTokens > 7) throw new ArgumentOutOfRangeException(nameof(draftTokens));
        if (!Enum.IsDefined(headPolicy) || (headPolicy == CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist) != (shortlist != null))
            throw new ArgumentException("Select the full target head or the original mapped shortlist explicitly.");
        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Count != 5 || layers.Any(x => x == null)) throw new ArgumentException("Expected five original draft layers.");
        ArgumentException.ThrowIfNullOrWhiteSpace(provenance);
        DraftTokens = draftTokens; HeadPolicy = headPolicy; Shortlist = shortlist;
        FeatureProjection = Projection(feature, 5120, 25600);
        ContextNorm = Plain(contextNorm, 5120); FinalNorm = Plain(finalNorm, 5120);
        Layers = Array.AsReadOnly(layers.ToArray());
        SelectorHidden = Projection(selectorHidden, 256, 5120);
        SelectorPredecessor = Projection(selectorPredecessor, 248320, 256);
        SelectorSuccessor = Projection(selectorSuccessor, 248320, 256); Provenance = provenance;
    }
    internal static CudaNinferProjectionDesc Projection(CudaNinferProjectionDesc projection, int rows, int columns)
    {
        ArgumentNullException.ThrowIfNull(projection);
        if (projection.Rows != rows || projection.Columns != columns || projection.Format != CudaNinferProjectionFormat.NvFp4Planes
            || projection.AllowA4 || BitConverter.SingleToInt32Bits(projection.InputScaleDivisor) != 0x3f800000)
            throw new ArgumentException("Original draft projections require their exact NVFP4 shape and A16-only policy without an activation divisor.");
        return projection;
    }
    internal static float[] Plain(float[] values, int length)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != length || values.Any(x => !float.IsFinite(x) || (BitConverter.SingleToInt32Bits(x) & 0xffff) != 0))
            throw new ArgumentException("Original draft parameters must be finite, exactly BF16-representable and correctly sized.");
        // Stored BF16 values are widened exactly, never offset by +1 or rounded again here.
        return (float[])values.Clone();
    }
}
