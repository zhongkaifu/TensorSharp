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
public enum NinferQwen38ParameterSemantics { None, RawALogRequiresNegativeExpf }
public enum NinferQwen38TensorRole
{
    TokenEmbedding, FinalNorm, OutputHead, InputNorm, PostAttentionNorm,
    GdnInput, GdnAlphaBeta, GdnALog, GdnDtBias, GdnConvolution, GdnNorm,
    AttentionInput, QueryNorm, KeyNorm, OutputProjection, GateUp, Down
}

/// <summary>One original reader-owned object with its semantic role, use policy and calibration (no converted bytes).</summary>
public sealed class NinferQwen38TensorBinding
{
    internal object PlanOwner { get; }
    /// <summary>The whole stored object (fused projections cover several logical bindings).</summary>
    public NinferTensorDescriptor Descriptor { get; }
    /// <summary>Logical binding name (the first part for fused objects).</summary>
    public string Name { get; }
    public NinferQwen38TensorRole Role { get; }
    public NinferQwen38NormConvention NormConvention { get; }
    public NinferQwen38RowOrder RowOrder { get; }
    public NinferQwen38ParameterSemantics ParameterSemantics { get; }
    /// <summary>Activation representations every use of this object permits.</summary>
    public NinferActivationPolicy Policy { get; }
    /// <summary>FP32 activation divisor of an A4 use (shared by the fused parts), null otherwise.</summary>
    public NinferTensorDescriptor? InputScaleDivisor { get; }
    public float? WeightDivisor => Descriptor.WeightDivisor;
    public uint? WeightDivisorBits => Descriptor.WeightDivisorBits;

    internal NinferQwen38TensorBinding(object owner, string name, NinferTensorDescriptor descriptor, NinferQwen38TensorRole role,
        NinferQwen38NormConvention norm, NinferQwen38RowOrder rows, NinferQwen38ParameterSemantics parameter,
        NinferActivationPolicy policy, NinferTensorDescriptor? inputScaleDivisor)
    {
        PlanOwner = owner; Name = name; Descriptor = descriptor; Role = role; NormConvention = norm;
        RowOrder = rows; ParameterSemantics = parameter; Policy = policy; InputScaleDivisor = inputScaleDivisor;
    }
}

public sealed class NinferQwen38LayerPlan
{
    public int Index { get; }
    public NinferQwen38LayerKind Kind { get; }
    public NinferQwen38TensorBinding InputNorm { get; internal set; } = null!;
    public NinferQwen38TensorBinding PostAttentionNorm { get; internal set; } = null!;
    public NinferQwen38TensorBinding InputProjection { get; internal set; } = null!;
    public NinferQwen38TensorBinding OutputProjection { get; internal set; } = null!;
    public NinferQwen38TensorBinding GateUp { get; internal set; } = null!;
    public NinferQwen38TensorBinding Down { get; internal set; } = null!;
    public NinferQwen38TensorBinding? AlphaBeta { get; internal set; }
    public NinferQwen38TensorBinding? ALog { get; internal set; }
    public NinferQwen38TensorBinding? DtBias { get; internal set; }
    public NinferQwen38TensorBinding? Convolution { get; internal set; }
    public NinferQwen38TensorBinding? GdnNorm { get; internal set; }
    public NinferQwen38TensorBinding? QueryNorm { get; internal set; }
    public NinferQwen38TensorBinding? KeyNorm { get; internal set; }
    internal NinferQwen38LayerPlan(int index)
    {
        Index = index;
        Kind = (index + 1) % 4 == 0 ? NinferQwen38LayerKind.GroupedQueryAttention : NinferQwen38LayerKind.GatedDeltaNet;
    }
}

/// <summary>
/// Immutable binding of the NInfer v3 Qwen3.8-27B text model (Neroued/Qwen3.8-27B-nvfp4-NInfer,
/// recipe qwen3_8_27b_nvfp4): FP8 row-scaled embedding, head, attention/GDN input and output
/// projections; NVFP4 or FP8 MLP per layer; BF16/FP32 control parameters. Fused objects
/// (GDN Q|K|V|Z, attention Q|K|gate|V, gate|up, a|b) are checked part by part. No bytes are
/// converted here; the caller owns and must keep the reader open.
/// </summary>
public sealed class NinferQwen38TargetPlan
{
    public const string ModelId = "qwen3.8-27b";
    public const string BindingVersion = "ninfer-v3-qwen38-target-v1";
    public const int LayerCount = 64, HiddenSize = 5120, VocabularySize = 248320, IntermediateSize = 17408;
    public const int AttentionQueryHeads = 24, AttentionKeyValueHeads = 4, AttentionHeadDimension = 256;
    public const int GdnQueryKeyHeads = 16, GdnValueHeads = 48, GdnHeadDimension = 128;
    public const int ConvolutionWidth = 4, RotaryDimension = 64;
    public const float RopeTheta = 10000000f, RmsEpsilon = 1e-6f;
    private readonly NinferArtifactReader _reader;
    private readonly object _owner = new();
    public NinferArtifactIdentity Identity => _reader.Identity;
    public string ArtifactPath => _reader.FullPath;
    public string DirectorySha256 => _reader.DirectorySha256;
    public IReadOnlyList<NinferQwen38TensorBinding> Tensors { get; }
    public IReadOnlyList<NinferQwen38LayerPlan> Layers { get; }
    public NinferQwen38TensorBinding TokenEmbedding { get; }
    public NinferQwen38TensorBinding FinalNorm { get; }
    public NinferQwen38TensorBinding OutputHead { get; }

    private NinferQwen38TargetPlan(NinferArtifactReader reader)
    {
        _reader = reader;
        var all = new List<NinferQwen38TensorBinding>();
        NinferQwen38TensorBinding Add(NinferQwen38TensorBinding b) { all.Add(b); return b; }
        TokenEmbedding = Add(Matrix("text/token_embedding", NinferQwen38TensorRole.TokenEmbedding, VocabularySize, HiddenSize,
            NinferQwen38RowOrder.Unchanged, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
        OutputHead = Add(Matrix("text/output_head", NinferQwen38TensorRole.OutputHead, VocabularySize, HiddenSize,
            NinferQwen38RowOrder.Unchanged, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
        FinalNorm = Add(Vector("text/final_norm", NinferQwen38TensorRole.FinalNorm, NinferTensorFormat.BFloat16, HiddenSize, NinferQwen38NormConvention.StoredUnitOffset));
        var layers = new NinferQwen38LayerPlan[LayerCount];
        for (int i = 0; i < LayerCount; i++)
        {
            string p = $"text/layers/{i}/";
            var l = layers[i] = new NinferQwen38LayerPlan(i);
            l.InputNorm = Add(Vector(p + "input_norm", NinferQwen38TensorRole.InputNorm, NinferTensorFormat.BFloat16, HiddenSize, NinferQwen38NormConvention.StoredUnitOffset));
            l.PostAttentionNorm = Add(Vector(p + "post_attention_norm", NinferQwen38TensorRole.PostAttentionNorm, NinferTensorFormat.BFloat16, HiddenSize, NinferQwen38NormConvention.StoredUnitOffset));
            if (l.Kind == NinferQwen38LayerKind.GatedDeltaNet)
            {
                l.InputProjection = Add(Fused(new[] { (p + "gdn/query", 2048), (p + "gdn/key", 2048), (p + "gdn/value", 6144), (p + "gdn/z", 6144) },
                    NinferQwen38TensorRole.GdnInput, NinferQwen38RowOrder.GdnQueryKeyValueZ, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
                l.AlphaBeta = Add(Fused(new[] { (p + "gdn/a_projection", 48), (p + "gdn/b_projection", 48) },
                    NinferQwen38TensorRole.GdnAlphaBeta, NinferQwen38RowOrder.GdnAlphaThenBeta, NinferTensorFormat.BFloat16));
                l.ALog = Add(Vector(p + "gdn/a_log", NinferQwen38TensorRole.GdnALog, NinferTensorFormat.Float32, GdnValueHeads,
                    parameter: NinferQwen38ParameterSemantics.RawALogRequiresNegativeExpf));
                l.DtBias = Add(Vector(p + "gdn/dt_bias", NinferQwen38TensorRole.GdnDtBias, NinferTensorFormat.Float32, GdnValueHeads));
                l.Convolution = Add(Plain(p + "gdn/convolution", NinferQwen38TensorRole.GdnConvolution, NinferTensorFormat.BFloat16,
                    new ulong[] { ConvolutionWidth, 10240 }, NinferQwen38RowOrder.ConvTapThenChannel));
                l.GdnNorm = Add(Vector(p + "gdn/norm", NinferQwen38TensorRole.GdnNorm, NinferTensorFormat.BFloat16, GdnHeadDimension, NinferQwen38NormConvention.StoredPlain));
                l.OutputProjection = Add(Matrix(p + "gdn/output", NinferQwen38TensorRole.OutputProjection, HiddenSize, 6144,
                    NinferQwen38RowOrder.Unchanged, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
            }
            else
            {
                l.InputProjection = Add(Fused(new[] { (p + "attention/query", 6144), (p + "attention/key", 1024), (p + "attention/gate", 6144), (p + "attention/value", 1024) },
                    NinferQwen38TensorRole.AttentionInput, NinferQwen38RowOrder.AttentionQueryKeyGateValue, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
                l.QueryNorm = Add(Vector(p + "attention/query_norm", NinferQwen38TensorRole.QueryNorm, NinferTensorFormat.BFloat16, AttentionHeadDimension, NinferQwen38NormConvention.StoredUnitOffset));
                l.KeyNorm = Add(Vector(p + "attention/key_norm", NinferQwen38TensorRole.KeyNorm, NinferTensorFormat.BFloat16, AttentionHeadDimension, NinferQwen38NormConvention.StoredUnitOffset));
                l.OutputProjection = Add(Matrix(p + "attention/output", NinferQwen38TensorRole.OutputProjection, HiddenSize, 6144,
                    NinferQwen38RowOrder.Unchanged, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
            }
            l.GateUp = Add(Fused(new[] { (p + "mlp/gate", IntermediateSize), (p + "mlp/up", IntermediateSize) },
                NinferQwen38TensorRole.GateUp, NinferQwen38RowOrder.MlpGateThenUp, NinferTensorFormat.NvFp4, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
            l.Down = Add(Matrix(p + "mlp/down", NinferQwen38TensorRole.Down, HiddenSize, IntermediateSize,
                NinferQwen38RowOrder.Unchanged, NinferTensorFormat.NvFp4, NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale));
        }
        Tensors = all.AsReadOnly();
        Layers = Array.AsReadOnly(layers);
        var objects = new HashSet<string>(StringComparer.Ordinal);
        foreach (var binding in all) Require(objects.Add(binding.Descriptor.Name), "A target object is bound twice: " + binding.Descriptor.Name);
    }

    public static NinferQwen38TargetPlan Create(NinferArtifactReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        reader.ReadRange(0, Span<byte>.Empty); // Real disposed-reader check; no payload read.
        if (reader.Identity.ModelId != ModelId)
            throw new NotSupportedException($"Unsupported NInfer model '{reader.Identity.ModelId}'; expected {ModelId}.");
        return new(reader);
    }

    public void ReadTensorRange(NinferQwen38TensorBinding tensor, ulong relativeOffset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (!ReferenceEquals(tensor.PlanOwner, _owner)) throw new ArgumentException("Binding belongs to another target plan.", nameof(tensor));
        _reader.ReadObjectRange(tensor.Descriptor, relativeOffset, destination);
    }

    /// <summary>Declared artifact head grouping.</summary>
    public static int GdnQueryKeyHead(int valueHead)
    {
        if ((uint)valueHead >= GdnValueHeads) throw new ArgumentOutOfRangeException(nameof(valueHead));
        return valueHead / (GdnValueHeads / GdnQueryKeyHeads);
    }

    private NinferQwen38TensorBinding Vector(string name, NinferQwen38TensorRole role, NinferTensorFormat format, int width,
        NinferQwen38NormConvention norm = NinferQwen38NormConvention.NotNorm,
        NinferQwen38ParameterSemantics parameter = NinferQwen38ParameterSemantics.None)
        => Plain(name, role, format, new[] { (ulong)width }, NinferQwen38RowOrder.Unchanged, norm, parameter);

    private NinferQwen38TensorBinding Plain(string name, NinferQwen38TensorRole role, NinferTensorFormat format, ulong[] shape,
        NinferQwen38RowOrder rows, NinferQwen38NormConvention norm = NinferQwen38NormConvention.NotNorm,
        NinferQwen38ParameterSemantics parameter = NinferQwen38ParameterSemantics.None)
    {
        var b = _reader.GetBinding(name);
        Require(b.IsWholeObject && b.Object.Format == format && b.Object.Shape.SequenceEqual(shape), $"{name}: expected whole {format} [{string.Join(",", shape)}].");
        return new(_owner, name, b.Object, role, norm, rows, parameter, NinferActivationPolicy.A16Only, null);
    }

    private NinferQwen38TensorBinding Matrix(string name, NinferQwen38TensorRole role, int rows, int columns,
        NinferQwen38RowOrder order, params NinferTensorFormat[] formats)
        => Fused(new[] { (name, rows) }, role, order, formats);

    /// <summary>Consecutive row ranges that must exactly tile one stored [rows, columns] object.</summary>
    private NinferQwen38TensorBinding Fused((string Name, int Rows)[] parts, NinferQwen38TensorRole role,
        NinferQwen38RowOrder order, params NinferTensorFormat[] formats)
    {
        NinferTensorDescriptor? tensor = null;
        int next = 0;
        NinferActivationPolicy? policy = null;
        NinferTensorDescriptor? divisor = null;
        float? divisorValue = null;
        foreach (var (name, rows) in parts)
        {
            var b = _reader.GetBinding(name);
            tensor ??= b.Object;
            Require(ReferenceEquals(tensor, b.Object), name + ": fused parts must share one stored object.");
            var (first, count) = b.RowRange;
            Require(first == next && count == rows, $"{name}: expected rows [{next},{next + rows}), found [{first},{first + count}).");
            next += rows;
            var uses = _reader.UsesOf(name);
            // The embedding is a gather, not a projection; every projection must declare its uses.
            Require(uses.Count > 0 || role == NinferQwen38TensorRole.TokenEmbedding, name + ": parameter has no use.");
            foreach (var use in uses)
            {
                policy ??= use.Policy;
                Require(policy == use.Policy, name + ": fused parts disagree on the activation policy.");
                if (use.ActivationInputDivisor is { } d)
                {
                    float value = _reader.ReadScalar(d);
                    Require(float.IsFinite(value) && value > 0, name + ": activation divisor must be positive and finite.");
                    // One activation quantization serves the fused projection, so the divisors must agree.
                    Require(divisorValue == null || divisorValue == value, name + ": fused parts carry different activation divisors.");
                    divisor ??= d; divisorValue = value;
                }
            }
        }
        Require(tensor!.Shape.Count == 2 && tensor.Shape[0] == (ulong)next && tensor.Shape[1] == (ulong)HiddenSizeFor(role),
            $"{parts[0].Name}: stored object shape [{string.Join(",", tensor.Shape)}] differs from the expected [{next},{HiddenSizeFor(role)}].");
        Require(formats.Contains(tensor.Format), $"{parts[0].Name}: unsupported format {tensor.Format}.");
        bool nvfp4 = tensor.Format == NinferTensorFormat.NvFp4;
        Require(!nvfp4 || (policy == NinferActivationPolicy.AllowA4 && divisor != null), parts[0].Name + ": NVFP4 projections need an A4 use with an activation divisor.");
        Require(nvfp4 || divisor == null, parts[0].Name + ": only NVFP4 uses carry an activation divisor.");
        return new(_owner, parts[0].Name, tensor, role, NinferQwen38NormConvention.NotNorm, order,
            NinferQwen38ParameterSemantics.None, policy ?? NinferActivationPolicy.A16Only, divisor);
    }

    private static int HiddenSizeFor(NinferQwen38TensorRole role) => role switch
    {
        NinferQwen38TensorRole.OutputProjection => 6144,
        NinferQwen38TensorRole.Down => IntermediateSize,
        _ => HiddenSize
    };

    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
}
