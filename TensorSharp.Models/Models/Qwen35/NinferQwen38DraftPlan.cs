// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

/// <summary>Original draft parameter semantics, distinct from the target's offset norms and attention row order.</summary>
public sealed class NinferQwen38DraftBinding
{
    internal object Owner { get; }
    public NinferTensorDescriptor Descriptor { get; }
    public NinferQwen38DraftRole Role { get; }
    public bool IsPlainNorm { get; }
    public NinferNvfp4RowMapping ProjectionRowMapping { get; }
    public bool AllowActivationQuantization => false;
    internal NinferQwen38DraftBinding(object owner, NinferTensorDescriptor descriptor, NinferQwen38DraftSpec spec)
        => (Owner, Descriptor, Role, IsPlainNorm, ProjectionRowMapping) = (owner, descriptor, spec.Role, spec.PlainNorm, spec.Mapping);
}

/// <summary>
/// Strict binding of the published 66-tensor DFlash2 module; optional shortlist remains an explicit caller choice.
/// Source: Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4, model.cu/dflash2_drafter.cu.
/// This plan neither attaches a drafter nor establishes native numerical or output-quality equivalence.
/// </summary>
public sealed class NinferQwen38DraftPlan
{
    public const string BindingVersion = "ninfer-qwen38-dflash2-original-v1";
    public const int LayerCount = 5, HiddenSize = 5120, FeatureWidth = 25600, IntermediateSize = 17408;
    public const int QueryHeads = 32, KeyValueHeads = 8, HeadDimension = 128;
    public const int RingPositions = 2048, ConvTaps = 2, ConvGroupSize = 16, ConvProjectionRows = 1280;
    public const int MaskToken = 248070, MaximumDraftTokens = 7, TopK = 16, SelectorRank = 256, ShortlistRows = 131072;
    public const int VocabularySize = NinferQwen38TargetPlan.VocabularySize;
    public const float RmsEpsilon = 1e-6f, RopeTheta = 10000000f;
    public const long KeyValueBytesPerSequence = (long)LayerCount * 2 * KeyValueHeads * RingPositions * HeadDimension * 2;
    public static IReadOnlyList<int> TargetOutputLayers { get; } = Array.AsReadOnly(new[] { 5, 19, 33, 47, 61 });
    private readonly object _owner = new();
    private readonly IReadOnlyDictionary<string, NinferQwen38DraftBinding> _byName;
    internal NinferArtifactReader Reader { get; }
    public IReadOnlyList<NinferQwen38DraftBinding> Tensors { get; }
    public bool HasShortlist => _byName.ContainsKey("text/draft_head");
    public bool InferenceSupported => false;
    public string DirectorySha256 => Reader.DirectorySha256;

    private NinferQwen38DraftPlan(NinferArtifactReader reader, IReadOnlyList<NinferQwen38DraftSpec> specs)
    {
        Reader = reader;
        var bindings = specs.Select(s => new NinferQwen38DraftBinding(_owner, (NinferTensorDescriptor)reader.GetObject(s.Name), s)).ToArray();
        Tensors = Array.AsReadOnly(bindings);
        _byName = new ReadOnlyDictionary<string, NinferQwen38DraftBinding>(bindings.ToDictionary(x => x.Descriptor.Name, StringComparer.Ordinal));
    }

    public static NinferQwen38DraftPlan Create(NinferArtifactReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        reader.ReadRange(0, Span<byte>.Empty);
        Require(!reader.Resources.Any(x => IsDraftName(x.Name)), "Draft parameters cannot be resources.");
        var specs = ValidateMetadata(reader.Identity, reader.Tensors.Select(NinferQwen38TensorMetadata.FromDescriptor), reader.PayloadStart, reader.FileBytes);
        return new(reader, specs);
    }

    public NinferQwen38DraftBinding GetTensor(string name) => _byName[name];
    public void ReadTensorRange(NinferQwen38DraftBinding binding, ulong offset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(binding);
        if (!ReferenceEquals(binding.Owner, _owner)) throw new ArgumentException("Binding belongs to another draft plan.", nameof(binding));
        Reader.ReadObjectRange(binding.Descriptor, offset, destination);
    }

    // Metadata DTOs do not manufacture readable descriptors or bypass reader ownership.
    internal static IReadOnlyList<NinferQwen38DraftSpec> ValidateMetadata(NinferArtifactIdentity identity,
        IEnumerable<NinferQwen38TensorMetadata> metadata, long payloadStart, long fileBytes)
    {
        if (identity.ModelId != NinferQwen38TargetPlan.ModelId || identity.WeightsId != NinferQwen38TargetPlan.WeightsId)
            throw new NotSupportedException("Unsupported original NInfer DFlash2 family/weights identity.");
        Require(payloadStart >= 16 && payloadStart % 4096 == 0 && fileBytes >= payloadStart, "Invalid payload bounds.");
        var all = new Dictionary<string, NinferQwen38TensorMetadata>(StringComparer.Ordinal);
        foreach (var t in metadata) Require(all.TryAdd(t.Name, t), "Duplicate tensor: " + t.Name);
        bool head = all.ContainsKey("text/draft_head"), ids = all.ContainsKey("text/draft_head_token_ids");
        Require(head == ids, "The shortlist head and token map must appear together.");
        var specs = Specifications(head); var expected = specs.ToDictionary(x => x.Name, StringComparer.Ordinal);
        var selected = all.Values.Where(x => IsDraftName(x.Name)).ToArray();
        Require(selected.Length == specs.Count && selected.All(x => expected.ContainsKey(x.Name)), "Unexpected or incomplete original DFlash2 namespace.");
        foreach (var spec in specs)
        {
            Require(all.TryGetValue(spec.Name, out var t), "Missing draft tensor: " + spec.Name);
            Require(t!.Format == spec.Format && t.Layout == spec.Layout && t.Shape.SequenceEqual(spec.Shape), "Draft format/layout/shape differs: " + spec.Name);
            var geometry = Geometry(spec);
            Require(t.Geometry == geometry && t.ByteLength == geometry.EncodedBytes, "Draft encoded geometry differs: " + spec.Name);
            Require(t.PayloadOffset % 256 == 0 && t.AbsoluteOffset >= payloadStart && t.AbsoluteOffset <= fileBytes
                && (ulong)(t.AbsoluteOffset - payloadStart) == t.PayloadOffset && t.ByteLength <= (ulong)(fileBytes - t.AbsoluteOffset), "Draft original range differs: " + spec.Name);
            if (spec.Format == NinferTensorFormat.NvFp4)
                Require(t.WeightDivisor is float d && float.IsFinite(d) && d > 0 && t.WeightDivisorBits is uint bits
                    && bits == unchecked((uint)BitConverter.SingleToInt32Bits(d)), "Draft weight divisor bits were not preserved: " + spec.Name);
            else Require(t.WeightDivisor == null && t.WeightDivisorBits == null, "Unexpected draft divisor: " + spec.Name);
        }
        ulong end = 0;
        foreach (var t in selected.OrderBy(x => x.PayloadOffset))
        {
            Require(t.PayloadOffset >= end, "Draft tensor ranges overlap: " + t.Name);
            end = checked(t.PayloadOffset + t.ByteLength);
        }
        return specs;
    }

    private static bool IsDraftName(string name) => name.StartsWith("dflash2/", StringComparison.Ordinal)
        || name.StartsWith("text/draft_", StringComparison.Ordinal);

    private static IReadOnlyList<NinferQwen38DraftSpec> Specifications(bool shortlist)
    {
        var result = new List<NinferQwen38DraftSpec>();
        void Add(string name, NinferQwen38DraftRole role, NinferTensorFormat format, ulong[] shape,
            bool norm = false, NinferNvfp4RowMapping mapping = NinferNvfp4RowMapping.Identity)
            => result.Add(new(name, role, format, format == NinferTensorFormat.NvFp4 ? NinferTensorLayout.BlockScaleK16M128x4V1
                : format == NinferTensorFormat.Q4G64Float16Scale ? NinferTensorLayout.RowSplitK128V1 : NinferTensorLayout.ContiguousLittleEndianV1,
                Array.AsReadOnly(shape), norm, mapping));
        void P(string name, NinferQwen38DraftRole role, ulong n, ulong k, bool gateUp = false)
            => Add(name, role, NinferTensorFormat.NvFp4, new[] { n, k }, mapping: gateUp ? NinferNvfp4RowMapping.GateUpRows8 : NinferNvfp4RowMapping.Identity);
        void Norm(string name, NinferQwen38DraftRole role, ulong n) => Add(name, role, NinferTensorFormat.BFloat16, new[] { n }, true);
        P("dflash2/feature_projection", NinferQwen38DraftRole.FeatureProjection, HiddenSize, FeatureWidth);
        Norm("dflash2/context_norm", NinferQwen38DraftRole.ContextNorm, HiddenSize);
        for (int i = 0; i < LayerCount; i++)
        {
            string p = $"dflash2/layers/{i}/";
            Norm(p + "input_norm", NinferQwen38DraftRole.InputNorm, HiddenSize);
            P(p + "attention/query_key_value", NinferQwen38DraftRole.QueryKeyValue, 6144, HiddenSize);
            Norm(p + "attention/query_norm", NinferQwen38DraftRole.QueryNorm, HeadDimension);
            Norm(p + "attention/key_norm", NinferQwen38DraftRole.KeyNorm, HeadDimension);
            P(p + "attention/output", NinferQwen38DraftRole.AttentionOutput, HiddenSize, 4096);
            Add(p + "attention/conv_base", NinferQwen38DraftRole.AttentionConvBase, NinferTensorFormat.BFloat16, new ulong[] { 2, 2, HiddenSize });
            P(p + "attention/conv_proj", NinferQwen38DraftRole.AttentionConvProjection, ConvProjectionRows, HiddenSize);
            Norm(p + "post_attention_norm", NinferQwen38DraftRole.PostAttentionNorm, HiddenSize);
            P(p + "mlp/gate_up", NinferQwen38DraftRole.GateUp, 34816, HiddenSize, true);
            P(p + "mlp/down", NinferQwen38DraftRole.Down, HiddenSize, IntermediateSize);
            Add(p + "mlp/conv_base", NinferQwen38DraftRole.MlpConvBase, NinferTensorFormat.BFloat16, new ulong[] { 2, 2, HiddenSize });
            P(p + "mlp/conv_proj", NinferQwen38DraftRole.MlpConvProjection, ConvProjectionRows, HiddenSize);
        }
        Norm("dflash2/final_norm", NinferQwen38DraftRole.FinalNorm, HiddenSize);
        P("dflash2/selector/hidden", NinferQwen38DraftRole.SelectorHidden, SelectorRank, HiddenSize);
        P("dflash2/selector/predecessor", NinferQwen38DraftRole.SelectorPredecessor, VocabularySize, SelectorRank);
        P("dflash2/selector/successor", NinferQwen38DraftRole.SelectorSuccessor, VocabularySize, SelectorRank);
        if (shortlist)
        {
            Add("text/draft_head", NinferQwen38DraftRole.ShortlistHead, NinferTensorFormat.Q4G64Float16Scale, new ulong[] { ShortlistRows, HiddenSize });
            Add("text/draft_head_token_ids", NinferQwen38DraftRole.ShortlistTokenIds, NinferTensorFormat.Int32, new ulong[] { ShortlistRows });
        }
        return result.AsReadOnly();
    }

    internal static NinferTensorGeometry Geometry(NinferQwen38DraftSpec spec)
    {
        ulong count = spec.Shape.Aggregate(1UL, (a, b) => checked(a * b));
        if (spec.Format == NinferTensorFormat.NvFp4)
            return new(count / 2 + count / 16 + 4, 256, LowPlaneBytes: count / 2, ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 16, DivisorOffset: count / 2 + count / 16);
        if (spec.Format == NinferTensorFormat.Q4G64Float16Scale)
            return new(count / 2 + count / 32, 256, PaddedK: HiddenSize, GroupSize: 64, LowPlaneBytes: count / 2,
                HighPlaneOffset: count / 2, HighPlaneBytes: 0, ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 32);
        return new(count * (spec.Format == NinferTensorFormat.BFloat16 ? 2UL : 4UL), 256);
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
}

internal sealed record NinferQwen38DraftSpec(string Name, NinferQwen38DraftRole Role, NinferTensorFormat Format,
    NinferTensorLayout Layout, IReadOnlyList<ulong> Shape, bool PlainNorm, NinferNvfp4RowMapping Mapping);
