// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Owns the original file while the native engine synchronously uploads its weights.</summary>
internal sealed class NinferQwen38ModelSource : IDisposable
{
    internal NinferArtifactReader Reader { get; }
    internal NinferQwen38TargetPlan Plan { get; }
    internal NinferQwen38Frontend Frontend { get; }
    internal string CacheIdentity { get; }
    private readonly List<IDisposable> _convertedSources = new();

    private NinferQwen38ModelSource(NinferArtifactReader reader)
    {
        Reader = reader;
        Plan = NinferQwen38TargetPlan.Create(reader);
        Frontend = NinferQwen38Frontend.FromReader(reader);
        // Separates ordinary modified/replaced deployments without advertising a
        // complete payload digest. Exact whole-file hashes remain external evidence.
        CacheIdentity = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new
        {
            reader.FullPath, reader.FileBytes, reader.FileLastWriteUtcTicks,
            reader.DirectorySha256, reader.PrefixSha256
        })))).ToLowerInvariant();
    }

    internal static NinferQwen38ModelSource Open(string path)
    {
        var reader = new NinferArtifactReader(path);
        try { return new(reader); }
        catch { reader.Dispose(); throw; }
    }

    // Detect the format before a GGUF parser or a CUDA allocator is constructed. The
    // suffix also routes damaged .ninfer files to the parser with the useful refusal.
    internal static bool IsNinferArtifact(string path)
        => NinferArtifactReader.IsNinferArtifact(path);

    internal static void ValidateExecutionRoute(BackendType backend, int tpDegree,
        ITensorParallelGroup? tpGroup, int layerSplitDegree, string? draftModelPath)
    {
        if (backend != BackendType.Cuda || tpDegree != 1 || tpGroup != null || layerSplitDegree != 1)
            throw new NotSupportedException("NInfer Qwen3.8 body inference requires --backend cuda on one device; tensor parallelism, layer split and ggml_cuda are not implemented for this format.");
        if (Environment.GetEnvironmentVariable("TS_Q35_NATIVE") == "0")
            throw new NotSupportedException("NInfer body inference requires the native CUDA engine; remove TS_Q35_NATIVE=0.");
        if (!string.IsNullOrWhiteSpace(draftModelPath))
            throw new NotSupportedException("An external draft model is not implemented for complete NInfer body inference. The artifact's target text model can run without a drafter.");
        if (Environment.GetEnvironmentVariable(Qwen35CudaExecutor.NinferHeadEnvironment) != null
            || Environment.GetEnvironmentVariable(Qwen35CudaExecutor.NinferHeadScopeEnvironment) != null)
            throw new NotSupportedException("The GGUF companion-head options cannot be combined with a complete NInfer body; unset TS_Q35_NINFER_HEAD and TS_Q35_NINFER_HEAD_SCOPE.");
    }

    internal float[] ReadFloats(NinferQwen38TensorBinding tensor)
    {
        if (tensor.Descriptor.Format is not (NinferTensorFormat.BFloat16 or NinferTensorFormat.Float32))
            throw new NotSupportedException("Only original BF16/FP32 parameters can be read as plain floats.");
        int bytes = checked((int)tensor.Descriptor.ByteLength);
        if (bytes > 1 << 20)
            throw new NotSupportedException("Plain parameter read exceeds the bounded 1 MiB vector/conv budget; projections must stream.");
        byte[] raw = new byte[bytes];
        Plan.ReadTensorRange(tensor, 0, raw);
        return DecodePlainFloats(raw, tensor.Descriptor.Format);
    }

    internal static float[] DecodePlainFloats(ReadOnlySpan<byte> raw, NinferTensorFormat format)
    {
        int width = format switch
        {
            NinferTensorFormat.BFloat16 => 2,
            NinferTensorFormat.Float32 => 4,
            _ => throw new NotSupportedException("Expected BF16 or FP32 original bytes.")
        };
        if (raw.Length % width != 0) throw new InvalidDataException("Truncated plain parameter.");
        var result = new float[raw.Length / width];
        for (int i = 0; i < result.Length; i++)
        {
            int bits = width == 2 ? BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)) << 16
                : BinaryPrimitives.ReadInt32LittleEndian(raw.Slice(i * 4, 4));
            result[i] = BitConverter.Int32BitsToSingle(bits);
            if (!float.IsFinite(result[i])) throw new InvalidDataException("Nonfinite plain model parameter.");
        }
        return result;
    }

    internal static float[] EffectiveNorm(float[] values, NinferQwen38NormConvention convention)
    {
        if (convention is not (NinferQwen38NormConvention.StoredPlain or NinferQwen38NormConvention.StoredUnitOffset))
            throw new InvalidDataException("Not a normalization tensor.");
        var result = (float[])values.Clone();
        if (convention == NinferQwen38NormConvention.StoredUnitOffset)
            for (int i = 0; i < result.Length; i++) result[i] += 1f;
        return result;
    }

    internal static float[] ChannelMajorConvolution(float[] tapMajor, int channels, int taps)
    {
        if (channels <= 0 || taps <= 0 || tapMajor.Length != checked(channels * taps))
            throw new ArgumentException("Convolution geometry differs.");
        var result = new float[tapMajor.Length];
        for (int t = 0; t < taps; t++)
            for (int c = 0; c < channels; c++) result[c * taps + t] = tapMajor[t * channels + c];
        return result;
    }

    internal static ModelConfig CreateConfig(NinferQwen38Frontend frontend) => new()
    {
        Architecture = "qwen35",
        NumLayers = NinferQwen38TargetPlan.LayerCount,
        HiddenSize = NinferQwen38TargetPlan.HiddenSize,
        IntermediateSize = NinferQwen38TargetPlan.IntermediateSize,
        NumHeads = NinferQwen38TargetPlan.AttentionQueryHeads,
        NumKVHeads = NinferQwen38TargetPlan.AttentionKeyValueHeads,
        KeyLength = NinferQwen38TargetPlan.AttentionHeadDimension,
        ValueLength = NinferQwen38TargetPlan.AttentionHeadDimension,
        Eps = NinferQwen38TargetPlan.RmsEpsilon,
        RopeBase = NinferQwen38TargetPlan.RopeTheta,
        VocabSize = NinferQwen38TargetPlan.VocabularySize,
        DeclaredContextLength = frontend.MaximumContextTokens,
        ChatTemplate = frontend.ChatTemplate,
        RecommendedSampling = new RecommendedSampling
        {
            Temperature = frontend.Generation.DoSample ? (float)frontend.Generation.Temperature : 0f,
            TopK = frontend.Generation.TopK,
            TopP = (float)frontend.Generation.TopP
        }
    };

    internal NinferNvfp4WeightSource ConvertedProjection(NinferQwen38TensorBinding tensor)
    {
        var source = new NinferNvfp4WeightSource(NinferNvfp4PlaneReader.FromTarget(Reader, tensor));
        _convertedSources.Add(source);
        return source;
    }

    public void Dispose()
    {
        foreach (var source in _convertedSources) source.Dispose();
        _convertedSources.Clear();
        Reader.Dispose();
    }
}
