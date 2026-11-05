// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public enum NinferQwen38DraftRole
{
    FeatureProjection, ContextNorm, InputNorm, QueryKeyValue, QueryNorm, KeyNorm,
    AttentionOutput, AttentionConvBase, AttentionConvProjection, PostAttentionNorm,
    GateUp, Down, MlpConvBase, MlpConvProjection, FinalNorm,
    SelectorHidden, SelectorPredecessor, SelectorSuccessor, ShortlistHead, ShortlistTokenIds
}

/// <summary>One stored DFlash2 object with its role; norms are plain (Qwen3 RMSNorm), never offset.</summary>
public sealed class NinferQwen38DraftBinding
{
    internal object Owner { get; }
    public string Name { get; }
    public NinferTensorDescriptor Descriptor { get; }
    public NinferQwen38DraftRole Role { get; }
    internal NinferQwen38DraftBinding(object owner, string name, NinferTensorDescriptor descriptor, NinferQwen38DraftRole role)
        => (Owner, Name, Descriptor, Role) = (owner, name, descriptor, role);
}

/// <summary>
/// The NInfer v3 DFlash2 companion of Qwen3.8-27B (z-lab/Qwen3.8-27B-DFlash2 weights in the
/// artifact): q8_g32 projections (Q|K|V and gate|up fused objects checked part by part), BF16
/// dynamic-convolution kernels/projections, BF16 candidate-selector codebooks, and the optional
/// indexed q4_g64 proposal head. Every use is A16Only. The caller owns the reader.
/// </summary>
public sealed class NinferQwen38DraftPlan
{
    public const string BindingVersion = "ninfer-v3-qwen38-dflash2-v1";
    public const int LayerCount = 5, HiddenSize = 5120, FeatureWidth = 25600, IntermediateSize = 17408;
    public const int QueryHeads = 32, KeyValueHeads = 8, HeadDimension = 128;
    public const int RingPositions = 2048, ConvTaps = 2, ConvGroupSize = 16, ConvProjectionRows = 1280;
    public const int MaskToken = 248070, MaximumDraftTokens = 7, TopK = 16, SelectorRank = 256, ShortlistRows = 131072;
    public const int VocabularySize = NinferQwen38TargetPlan.VocabularySize;
    public const float RmsEpsilon = 1e-6f, RopeTheta = 10000000f;
    public const long KeyValueBytesPerSequence = (long)LayerCount * 2 * KeyValueHeads * RingPositions * HeadDimension * 2;
    public static IReadOnlyList<int> TargetOutputLayers { get; } = Array.AsReadOnly(new[] { 5, 19, 33, 47, 61 });
    private readonly object _owner = new();
    private readonly Dictionary<string, NinferQwen38DraftBinding> _byName = new(StringComparer.Ordinal);
    internal NinferArtifactReader Reader { get; }
    public IReadOnlyList<NinferQwen38DraftBinding> Tensors { get; }
    public bool HasShortlist { get; }
    public string DirectorySha256 => Reader.DirectorySha256;

    private NinferQwen38DraftPlan(NinferArtifactReader reader)
    {
        Reader = reader;
        var all = new List<NinferQwen38DraftBinding>();
        void Add(string key, NinferQwen38DraftBinding b) { _byName.Add(key, b); all.Add(b); }
        void Norm(string name, NinferQwen38DraftRole role, ulong width) => Add(name, Whole(name, role, NinferTensorFormat.BFloat16, width));
        void Matrix(string name, NinferQwen38DraftRole role, NinferTensorFormat format, ulong rows, ulong columns)
            => Add(name, Whole(name, role, format, rows, columns));
        Matrix("dflash2/feature_projection", NinferQwen38DraftRole.FeatureProjection, NinferTensorFormat.W8G32Float16Scale, HiddenSize, FeatureWidth);
        Norm("dflash2/context_norm", NinferQwen38DraftRole.ContextNorm, HiddenSize);
        for (int i = 0; i < LayerCount; i++)
        {
            string p = $"dflash2/layers/{i}/";
            Norm(p + "input_norm", NinferQwen38DraftRole.InputNorm, HiddenSize);
            Add(p + "attention/query_key_value", Fused(new[] { (p + "attention/query", 4096), (p + "attention/key", 1024), (p + "attention/value", 1024) },
                NinferQwen38DraftRole.QueryKeyValue, HiddenSize));
            Require(SameRange(p + "attention/context_key", p + "attention/key") && SameRange(p + "attention/context_value", p + "attention/value"),
                p + ": context K/V must share the query-side K/V projection rows.");
            Norm(p + "attention/query_norm", NinferQwen38DraftRole.QueryNorm, HeadDimension);
            Norm(p + "attention/key_norm", NinferQwen38DraftRole.KeyNorm, HeadDimension);
            Matrix(p + "attention/output", NinferQwen38DraftRole.AttentionOutput, NinferTensorFormat.W8G32Float16Scale, HiddenSize, 4096);
            Add(p + "attention_conv/base_kernel", Whole(p + "attention_conv/base_kernel", NinferQwen38DraftRole.AttentionConvBase, NinferTensorFormat.BFloat16, 2, 2, HiddenSize));
            Matrix(p + "attention_conv/kernel_projection", NinferQwen38DraftRole.AttentionConvProjection, NinferTensorFormat.BFloat16, ConvProjectionRows, HiddenSize);
            Norm(p + "post_attention_norm", NinferQwen38DraftRole.PostAttentionNorm, HiddenSize);
            Add(p + "mlp/gate_up", Fused(new[] { (p + "mlp/gate", IntermediateSize), (p + "mlp/up", IntermediateSize) },
                NinferQwen38DraftRole.GateUp, HiddenSize));
            Matrix(p + "mlp/down", NinferQwen38DraftRole.Down, NinferTensorFormat.W8G32Float16Scale, HiddenSize, IntermediateSize);
            Add(p + "mlp_conv/base_kernel", Whole(p + "mlp_conv/base_kernel", NinferQwen38DraftRole.MlpConvBase, NinferTensorFormat.BFloat16, 2, 2, HiddenSize));
            Matrix(p + "mlp_conv/kernel_projection", NinferQwen38DraftRole.MlpConvProjection, NinferTensorFormat.BFloat16, ConvProjectionRows, HiddenSize);
        }
        Norm("dflash2/final_norm", NinferQwen38DraftRole.FinalNorm, HiddenSize);
        Matrix("dflash2/candidate_selector/hidden_projection", NinferQwen38DraftRole.SelectorHidden, NinferTensorFormat.BFloat16, SelectorRank, HiddenSize);
        Matrix("dflash2/candidate_selector/predecessor_codebook", NinferQwen38DraftRole.SelectorPredecessor, NinferTensorFormat.BFloat16, VocabularySize, SelectorRank);
        Matrix("dflash2/candidate_selector/successor_codebook", NinferQwen38DraftRole.SelectorSuccessor, NinferTensorFormat.BFloat16, VocabularySize, SelectorRank);
        HasShortlist = reader.Bindings.ContainsKey("proposal/head");
        if (HasShortlist)
        {
            Matrix("proposal/head", NinferQwen38DraftRole.ShortlistHead, NinferTensorFormat.Q4G64Float16Scale, ShortlistRows, HiddenSize);
            Add("proposal/token_ids", Whole("proposal/token_ids", NinferQwen38DraftRole.ShortlistTokenIds, NinferTensorFormat.Int32, ShortlistRows));
        }
        Tensors = all.AsReadOnly();
    }

    public static NinferQwen38DraftPlan Create(NinferArtifactReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        reader.ReadRange(0, Span<byte>.Empty);
        if (reader.Identity.ModelId != NinferQwen38TargetPlan.ModelId)
            throw new NotSupportedException("Unsupported NInfer DFlash2 model: " + reader.Identity.ModelId);
        if (!reader.ComponentConfigs.ContainsKey("dflash2"))
            throw new NotSupportedException("The NInfer artifact carries no DFlash2 component.");
        return new(reader);
    }

    public NinferQwen38DraftBinding GetTensor(string name) => _byName[name];
    public void ReadTensorRange(NinferQwen38DraftBinding binding, ulong offset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(binding);
        if (!ReferenceEquals(binding.Owner, _owner)) throw new ArgumentException("Binding belongs to another draft plan.", nameof(binding));
        Reader.ReadObjectRange(binding.Descriptor, offset, destination);
    }

    private NinferQwen38DraftBinding Whole(string name, NinferQwen38DraftRole role, NinferTensorFormat format, params ulong[] shape)
    {
        var b = Reader.GetBinding(name);
        Require(b.IsWholeObject && b.Object.Format == format && b.Object.Shape.SequenceEqual(shape),
            $"{name}: expected whole {format} [{string.Join(",", shape)}], found {b.Object.Format} [{string.Join(",", b.Object.Shape)}].");
        foreach (var use in Reader.UsesOf(name)) Require(use.Policy == NinferActivationPolicy.A16Only, name + ": draft uses are A16-only.");
        return new(_owner, name, b.Object, role);
    }

    private NinferQwen38DraftBinding Fused((string Name, int Rows)[] parts, NinferQwen38DraftRole role, int columns)
    {
        NinferTensorDescriptor? tensor = null;
        int next = 0;
        foreach (var (name, rows) in parts)
        {
            var b = Reader.GetBinding(name);
            tensor ??= b.Object;
            Require(ReferenceEquals(tensor, b.Object), name + ": fused parts must share one stored object.");
            var (first, count) = b.RowRange;
            Require(first == next && count == rows, $"{name}: expected rows [{next},{next + rows}).");
            next += rows;
            foreach (var use in Reader.UsesOf(name)) Require(use.Policy == NinferActivationPolicy.A16Only, name + ": draft uses are A16-only.");
        }
        Require(tensor!.Format == NinferTensorFormat.W8G32Float16Scale && tensor.Shape.Count == 2
            && tensor.Shape[0] == (ulong)next && tensor.Shape[1] == (ulong)columns, parts[0].Name + ": expected a q8_g32 fused object.");
        return new(_owner, parts[0].Name, tensor, role);
    }

    private bool SameRange(string a, string b)
    {
        var x = Reader.GetBinding(a); var y = Reader.GetBinding(b);
        return ReferenceEquals(x.Object, y.Object) && x.ElementBegin == y.ElementBegin && x.ElementEnd == y.ElementEnd;
    }

    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
}
