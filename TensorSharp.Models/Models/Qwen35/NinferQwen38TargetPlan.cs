// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public enum NinferQwen38LayerKind { GatedDeltaNet, GroupedQueryAttention }
public enum NinferQwen38NormConvention { NotNorm, StoredUnitOffset, StoredPlain }
public enum NinferQwen38RowOrder
{
    Unchanged, GdnQueryKeyValueZ, GdnAlphaThenBeta,
    AttentionQueryKeyGateValue, MlpGateThenUp, ConvTapThenChannel
}
public enum NinferQwen38ParameterSemantics { None, RawALogRequiresNegativeExpf, CalibratedActivationDivisor }
public enum NinferQwen38ContentValidation { NotRead, PositiveFiniteWeightDivisorOnly }
public enum NinferQwen38TensorRole
{
    TokenEmbedding, FinalNorm, OutputHead, InputNorm, PostAttentionNorm,
    GdnInput, GdnAlphaBeta, GdnALog, GdnDtBias, GdnConvolution, GdnNorm,
    AttentionInput, QueryNorm, KeyNorm, OutputProjection, GateUp, Down, InputScaleDivisor
}

/// <summary>One original reader-owned tensor, with semantic declarations but no converted bytes.</summary>
public sealed class NinferQwen38TensorBinding
{
    internal object PlanOwner { get; }
    public NinferTensorDescriptor Descriptor { get; }
    public NinferQwen38TensorRole Role { get; }
    public NinferQwen38NormConvention NormConvention { get; }
    public NinferQwen38RowOrder RowOrder { get; }
    public NinferQwen38ParameterSemantics ParameterSemantics { get; }
    /// <summary>Separate FP32 calibration tensor. Its contents have NOT been read or validated by this plan.</summary>
    public NinferTensorDescriptor? InputScaleDivisor { get; }
    public NinferQwen38ContentValidation InputScaleDivisorContentValidation => NinferQwen38ContentValidation.NotRead;
    public float? WeightDivisor => Descriptor.WeightDivisor;
    public uint? WeightDivisorBits => Descriptor.WeightDivisorBits;
    public NinferQwen38ContentValidation ContentValidation => Descriptor.Format == NinferTensorFormat.NvFp4
        ? NinferQwen38ContentValidation.PositiveFiniteWeightDivisorOnly : NinferQwen38ContentValidation.NotRead;

    internal NinferQwen38TensorBinding(object owner, NinferTensorDescriptor descriptor,
        NinferQwen38TensorSpec spec, NinferTensorDescriptor? inputScaleDivisor)
    {
        PlanOwner = owner; Descriptor = descriptor; Role = spec.Role; NormConvention = spec.Norm;
        RowOrder = spec.Rows; ParameterSemantics = spec.Parameter; InputScaleDivisor = inputScaleDivisor;
    }
}

public sealed class NinferQwen38LayerPlan
{
    public int Index { get; }
    public NinferQwen38LayerKind Kind { get; }
    public IReadOnlyList<NinferQwen38TensorBinding> Tensors { get; }
    public NinferQwen38TensorBinding InputNorm { get; }
    public NinferQwen38TensorBinding PostAttentionNorm { get; }
    public NinferQwen38TensorBinding InputProjection { get; }
    public NinferQwen38TensorBinding OutputProjection { get; }
    public NinferQwen38TensorBinding GateUp { get; }
    public NinferQwen38TensorBinding Down { get; }
    public NinferQwen38TensorBinding? AlphaBeta { get; }
    public NinferQwen38TensorBinding? ALog { get; }
    public NinferQwen38TensorBinding? DtBias { get; }
    public NinferQwen38TensorBinding? Convolution { get; }
    public NinferQwen38TensorBinding? GdnNorm { get; }
    public NinferQwen38TensorBinding? QueryNorm { get; }
    public NinferQwen38TensorBinding? KeyNorm { get; }

    internal NinferQwen38LayerPlan(int index, IReadOnlyDictionary<string, NinferQwen38TensorBinding> tensors)
    {
        Index = index; Kind = (index + 1) % 4 == 0 ? NinferQwen38LayerKind.GroupedQueryAttention : NinferQwen38LayerKind.GatedDeltaNet;
        string p = $"text/layers/{index}/";
        NinferQwen38TensorBinding Get(string suffix) => tensors[p + suffix];
        InputNorm = Get("input_norm"); PostAttentionNorm = Get("post_attention_norm");
        GateUp = Get("mlp/gate_up"); Down = Get("mlp/down");
        if (Kind == NinferQwen38LayerKind.GatedDeltaNet)
        {
            InputProjection = Get("gdn/query_key_value_z"); OutputProjection = Get("gdn/output");
            AlphaBeta = Get("gdn/a_b_projection"); ALog = Get("gdn/a_log"); DtBias = Get("gdn/dt_bias");
            Convolution = Get("gdn/convolution"); GdnNorm = Get("gdn/norm");
        }
        else
        {
            InputProjection = Get("attention/query_key_gate_value"); OutputProjection = Get("attention/output");
            QueryNorm = Get("attention/query_norm"); KeyNorm = Get("attention/key_norm");
        }
        Tensors = Array.AsReadOnly(tensors.Values.Where(x => x.Descriptor.Name.StartsWith(p, StringComparison.Ordinal)).ToArray());
    }
}

/// <summary>
/// Immutable metadata binding for the known Qwen3.8-27B nvfp4full v2 family.
/// It does not enable inference, bind a tokenizer, convert/reorder weights, evaluate exp,
/// add norm offsets, or implement draft/vision. The caller owns and must keep the reader open.
/// </summary>
public sealed class NinferQwen38TargetPlan
{
    public const string ModelId = "qwen3.8-27b", WeightsId = "nvfp4full";
    public const string BindingVersion = "ninfer-qwen38-target-raw-v1";
    public const int LayerCount = 64, HiddenSize = 5120, VocabularySize = 248320, IntermediateSize = 17408;
    public const int AttentionQueryHeads = 24, AttentionKeyValueHeads = 4, AttentionHeadDimension = 256;
    public const int GdnQueryKeyHeads = 16, GdnValueHeads = 48, GdnHeadDimension = 128;
    public const int ConvolutionWidth = 4, RotaryDimension = 64;
    public const float RopeTheta = 10000000f, RmsEpsilon = 1e-6f;
    private readonly NinferArtifactReader _reader;
    private readonly object _owner = new();
    private readonly IReadOnlyDictionary<string, NinferQwen38TensorBinding> _byName;
    public NinferArtifactIdentity Identity => _reader.Identity;
    public string ArtifactPath => _reader.FullPath;
    public string DirectorySha256 => _reader.DirectorySha256;
    public IReadOnlyList<NinferQwen38TensorBinding> Tensors { get; }
    public IReadOnlyList<NinferQwen38LayerPlan> Layers { get; }
    /// <summary>Only original descriptors; no resource is implicitly read, parsed, extracted or bound.</summary>
    public IReadOnlyList<NinferResourceDescriptor> Resources { get; }
    public NinferQwen38TensorBinding TokenEmbedding => _byName["text/token_embedding"];
    public NinferQwen38TensorBinding FinalNorm => _byName["text/final_norm"];
    public NinferQwen38TensorBinding OutputHead => _byName["text/output_head"];
    public bool InferenceSupported => false;
    public bool TokenizerBound => false;
    public bool DraftBound => false;
    public bool VisionBound => false;
    /// <summary>Existing h%16 kernels are incompatible; preserve source head/K order with new h/3 kernels.</summary>
    public bool RequiresGroupedGdnKernel => true;
    public int TargetNvfp4WeightDivisorsValidated => Tensors.Count(x => x.Descriptor.Format == NinferTensorFormat.NvFp4);

    private NinferQwen38TargetPlan(NinferArtifactReader reader, IReadOnlyList<NinferQwen38TensorSpec> specs)
    {
        _reader = reader;
        var originals = reader.Tensors.ToDictionary(x => x.Name, StringComparer.Ordinal);
        var bindings = specs.Select(s => new NinferQwen38TensorBinding(_owner, originals[s.Name], s,
            s.InputScaleDivisorName == null ? null : originals[s.InputScaleDivisorName])).ToArray();
        _byName = new ReadOnlyDictionary<string, NinferQwen38TensorBinding>(bindings.ToDictionary(x => x.Descriptor.Name, StringComparer.Ordinal));
        Tensors = Array.AsReadOnly(bindings);
        Layers = Array.AsReadOnly(Enumerable.Range(0, LayerCount).Select(i => new NinferQwen38LayerPlan(i, _byName)).ToArray());
        Resources = Array.AsReadOnly(reader.Resources.ToArray());
    }

    public static NinferQwen38TargetPlan Create(NinferArtifactReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        reader.ReadRange(0, Span<byte>.Empty); // Real disposed-reader check; no payload read or fake descriptor owner.
        var specs = ValidateMetadata(reader.Identity, reader.Tensors.Select(NinferQwen38TensorMetadata.FromDescriptor), reader.PayloadStart, reader.FileBytes);
        return new(reader, specs);
    }

    public NinferQwen38TensorBinding GetTensor(string name) => _byName[name];
    public void ReadTensorRange(NinferQwen38TensorBinding tensor, ulong relativeOffset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (!ReferenceEquals(tensor.PlanOwner, _owner)) throw new ArgumentException("Binding belongs to another target plan.", nameof(tensor));
        _reader.ReadObjectRange(tensor.Descriptor, relativeOffset, destination);
    }
    public byte[] ReadResource(NinferResourceDescriptor resource, int maxBytes = 16 << 20) => _reader.ReadResource(resource, maxBytes);

    /// <summary>Declared artifact head grouping only, not a call into the incompatible current native kernels.</summary>
    public static int GdnQueryKeyHead(int valueHead)
    {
        if ((uint)valueHead >= GdnValueHeads) throw new ArgumentOutOfRangeException(nameof(valueHead));
        return valueHead / (GdnValueHeads / GdnQueryKeyHeads);
    }

    // CPU-only validation DTOs cannot manufacture reader-owned descriptors or a readable target plan.
    internal static IReadOnlyList<NinferQwen38TensorSpec> ValidateMetadata(NinferArtifactIdentity identity,
        IEnumerable<NinferQwen38TensorMetadata> metadata, long payloadStart, long fileBytes)
    {
        if (identity.ModelId != ModelId || identity.WeightsId != WeightsId)
            throw new NotSupportedException("Unsupported NInfer Qwen target family/weights identity.");
        Require(payloadStart >= 16 && payloadStart % 4096 == 0 && fileBytes >= payloadStart, "Invalid source payload/file bounds.");
        var all = new Dictionary<string, NinferQwen38TensorMetadata>(StringComparer.Ordinal);
        foreach (var tensor in metadata) Require(all.TryAdd(tensor.Name, tensor), "Duplicate tensor: " + tensor.Name);
        var specs = Specifications(); var expected = specs.ToDictionary(x => x.Name, StringComparer.Ordinal);
        var target = all.Where(x => x.Key.StartsWith("text/", StringComparison.Ordinal)
            && x.Key != "text/draft_head" && x.Key != "text/draft_head_token_ids").ToArray();
        Require(target.Length == 906 && target.All(x => expected.ContainsKey(x.Key)), "Expected exactly the complete 906-tensor target namespace.");
        foreach (var spec in specs)
        {
            Require(all.TryGetValue(spec.Name, out var tensor), "Missing target tensor: " + spec.Name);
            Require(tensor!.Format == spec.Format && tensor.Layout == spec.Layout && tensor.Shape.SequenceEqual(spec.Shape), "Tensor format/layout/shape differs: " + spec.Name);
            var geometry = ExpectedGeometry(spec);
            Require(tensor.Geometry == geometry && tensor.ByteLength == geometry.EncodedBytes, "Encoded geometry differs: " + spec.Name);
            Require(tensor.PayloadOffset % 256 == 0 && tensor.AbsoluteOffset >= payloadStart
                && (ulong)(tensor.AbsoluteOffset - payloadStart) == tensor.PayloadOffset
                && tensor.AbsoluteOffset <= fileBytes && tensor.ByteLength <= (ulong)(fileBytes - tensor.AbsoluteOffset), "Original source range differs: " + spec.Name);
            if (spec.Format == NinferTensorFormat.NvFp4)
            {
                Require(tensor.WeightDivisor is float d && float.IsFinite(d) && d > 0 && tensor.WeightDivisorBits is uint bits
                    && bits == unchecked((uint)BitConverter.SingleToInt32Bits(d)), "Weight divisor was not validated/preserved: " + spec.Name);
            }
            else Require(tensor.WeightDivisor == null && tensor.WeightDivisorBits == null, "Unexpected weight divisor: " + spec.Name);
        }
        ulong end = 0;
        foreach (var tensor in target.Select(x => x.Value).OrderBy(x => x.PayloadOffset))
        {
            Require(tensor.PayloadOffset >= end, "Target ranges overlap: " + tensor.Name);
            end = checked(tensor.PayloadOffset + tensor.ByteLength);
        }
        return specs;
    }

    private static IReadOnlyList<NinferQwen38TensorSpec> Specifications()
    {
        var specs = new List<NinferQwen38TensorSpec>();
        void Add(string name, NinferQwen38TensorRole role, NinferTensorFormat format, ulong[] shape,
            NinferQwen38NormConvention norm = NinferQwen38NormConvention.NotNorm,
            NinferQwen38RowOrder rows = NinferQwen38RowOrder.Unchanged,
            NinferQwen38ParameterSemantics parameter = NinferQwen38ParameterSemantics.None, string? input = null)
        {
            var layout = format == NinferTensorFormat.NvFp4 ? NinferTensorLayout.BlockScaleK16M128x4V1
                : format == NinferTensorFormat.W8G32Float16Scale ? NinferTensorLayout.RowSplitK128V1 : NinferTensorLayout.ContiguousLittleEndianV1;
            specs.Add(new(name, role, format, layout, Array.AsReadOnly(shape), norm, rows, parameter, input));
        }
        void Norm(string name, NinferQwen38TensorRole role, ulong width, bool offset = true) => Add(name, role, NinferTensorFormat.BFloat16,
            new[] { width }, offset ? NinferQwen38NormConvention.StoredUnitOffset : NinferQwen38NormConvention.StoredPlain);
        void Projection(string name, string divisorName, NinferQwen38TensorRole role, ulong rows, ulong k,
            bool bf16 = false, NinferQwen38RowOrder order = NinferQwen38RowOrder.Unchanged)
        {
            Add(name, role, bf16 ? NinferTensorFormat.BFloat16 : NinferTensorFormat.NvFp4, new[] { rows, k }, rows: order, input: bf16 ? null : divisorName);
            if (!bf16) Add(divisorName, NinferQwen38TensorRole.InputScaleDivisor, NinferTensorFormat.Float32, Array.Empty<ulong>(), parameter: NinferQwen38ParameterSemantics.CalibratedActivationDivisor);
        }
        Add("text/token_embedding", NinferQwen38TensorRole.TokenEmbedding, NinferTensorFormat.W8G32Float16Scale, new ulong[] { VocabularySize, HiddenSize });
        Norm("text/final_norm", NinferQwen38TensorRole.FinalNorm, HiddenSize);
        Add("text/output_head", NinferQwen38TensorRole.OutputHead, NinferTensorFormat.W8G32Float16Scale, new ulong[] { VocabularySize, HiddenSize });
        for (int i = 0; i < LayerCount; i++)
        {
            string p = $"text/layers/{i}/";
            Norm(p + "input_norm", NinferQwen38TensorRole.InputNorm, HiddenSize);
            Norm(p + "post_attention_norm", NinferQwen38TensorRole.PostAttentionNorm, HiddenSize);
            if ((i + 1) % 4 != 0)
            {
                Add(p + "gdn/a_log", NinferQwen38TensorRole.GdnALog, NinferTensorFormat.Float32, new ulong[] { 48 }, parameter: NinferQwen38ParameterSemantics.RawALogRequiresNegativeExpf);
                Add(p + "gdn/dt_bias", NinferQwen38TensorRole.GdnDtBias, NinferTensorFormat.Float32, new ulong[] { 48 });
                Add(p + "gdn/convolution", NinferQwen38TensorRole.GdnConvolution, NinferTensorFormat.BFloat16, new ulong[] { 4, 10240 }, rows: NinferQwen38RowOrder.ConvTapThenChannel);
                Add(p + "gdn/a_b_projection", NinferQwen38TensorRole.GdnAlphaBeta, NinferTensorFormat.BFloat16, new ulong[] { 96, HiddenSize }, rows: NinferQwen38RowOrder.GdnAlphaThenBeta);
                Projection(p + "gdn/query_key_value_z", p + "gdn/input_projection/input_scale_divisor", NinferQwen38TensorRole.GdnInput, 16384, HiddenSize, order: NinferQwen38RowOrder.GdnQueryKeyValueZ);
                Norm(p + "gdn/norm", NinferQwen38TensorRole.GdnNorm, 128, offset: false);
                Projection(p + "gdn/output", p + "gdn/output_projection/input_scale_divisor", NinferQwen38TensorRole.OutputProjection, HiddenSize, 6144, bf16: i == 4);
            }
            else
            {
                Projection(p + "attention/query_key_gate_value", p + "attention/input_projection/input_scale_divisor", NinferQwen38TensorRole.AttentionInput,
                    14336, HiddenSize, bf16: i <= 23, order: NinferQwen38RowOrder.AttentionQueryKeyGateValue);
                Norm(p + "attention/query_norm", NinferQwen38TensorRole.QueryNorm, 256);
                Norm(p + "attention/key_norm", NinferQwen38TensorRole.KeyNorm, 256);
                Projection(p + "attention/output", p + "attention/output_projection/input_scale_divisor", NinferQwen38TensorRole.OutputProjection, HiddenSize, 6144, bf16: i <= 7);
            }
            Projection(p + "mlp/gate_up", p + "mlp/gate_up_projection/input_scale_divisor", NinferQwen38TensorRole.GateUp, 34816, HiddenSize, order: NinferQwen38RowOrder.MlpGateThenUp);
            Projection(p + "mlp/down", p + "mlp/down_projection/input_scale_divisor", NinferQwen38TensorRole.Down, HiddenSize, IntermediateSize);
        }
        return specs.AsReadOnly();
    }

    private static NinferTensorGeometry ExpectedGeometry(NinferQwen38TensorSpec spec)
    {
        ulong count = 1;
        foreach (ulong dim in spec.Shape) count = checked(count * dim);
        if (spec.Format == NinferTensorFormat.NvFp4)
            return new(count / 2 + count / 16 + 4, 256, LowPlaneBytes: count / 2,
                ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 16, DivisorOffset: count / 2 + count / 16);
        if (spec.Format == NinferTensorFormat.W8G32Float16Scale)
            return new(count + count / 16, 256, PaddedK: spec.Shape[1], GroupSize: 32, LowPlaneBytes: count,
                HighPlaneOffset: count, HighPlaneBytes: 0, ScalePlaneOffset: count, ScalePlaneBytes: count / 16);
        return new(checked(count * (spec.Format == NinferTensorFormat.BFloat16 ? 2UL : 4UL)), 256);
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
}

internal sealed record NinferQwen38TensorSpec(string Name, NinferQwen38TensorRole Role, NinferTensorFormat Format,
    NinferTensorLayout Layout, IReadOnlyList<ulong> Shape, NinferQwen38NormConvention Norm,
    NinferQwen38RowOrder Rows, NinferQwen38ParameterSemantics Parameter, string? InputScaleDivisorName);

internal sealed record NinferQwen38TensorMetadata(string Name, NinferTensorFormat Format, NinferTensorLayout Layout,
    IReadOnlyList<ulong> Shape, ulong PayloadOffset, long AbsoluteOffset, ulong ByteLength,
    NinferTensorGeometry Geometry, float? WeightDivisor, uint? WeightDivisorBits)
{
    internal static NinferQwen38TensorMetadata FromDescriptor(NinferTensorDescriptor d) => new(d.Name, d.Format, d.Layout,
        d.Shape, d.PayloadOffset, d.AbsoluteOffset, d.ByteLength, d.Geometry, d.WeightDivisor, d.WeightDivisorBits);
}
