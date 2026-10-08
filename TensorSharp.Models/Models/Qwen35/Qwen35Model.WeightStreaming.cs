// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models;

public partial class Qwen35Model
{
    /// <summary>Small resident F32 normalization and recurrent parameters, excluded
    /// from the bounded quantized-weight staging budget. Activations and KV/state
    /// have their own lifetimes and are also outside this number.</summary>
    public long StreamingResidentParameterBytes { get; private set; }

    private void ValidateStreamingWeightConfiguration(string draftModelPath)
    {
        if (!HasStreamingWeights) return;
        StreamingResidentParameterBytes = ValidateStreamingWeightMetadata(
            Config.Architecture, _backend, GlobalTpDegree, LayerSplitDegree,
            _numExperts, _numNextnLayers, draftModelPath, _gguf.Tensors.Values);
        Console.WriteLine($"  File-backed Qwen35 weights: original Q8_0 projections, synchronous row tiles, " +
            $"no full-weight preload or fused weight graphs. Small resident F32 parameters: {StreamingResidentParameterBytes} bytes " +
            "(outside the quantized-weight staging budget; activations and KV/state are separate).");
    }

    internal static long ValidateStreamingWeightMetadata(string architecture, BackendType backend,
        int tensorParallelDegree, int layerSplitDegree, int experts, int mtpLayers,
        string draftModelPath, IEnumerable<GgufTensorInfo> tensors)
    {
        if (architecture != "qwen35" || backend != BackendType.GgmlCuda
            || tensorParallelDegree != 1 || layerSplitDegree != 1
            || experts != 0 || mtpLayers != 0 || !string.IsNullOrEmpty(draftModelPath))
            throw new NotSupportedException("File-backed Qwen35 weights require dense qwen35, one GGML CUDA rank, " +
                "and no MTP or external draft model. Tensor parallelism, layer splitting and other backends are unsupported.");

        const long maximumTensorBytes = 1L << 20;
        const long maximumResidentBytes = 32L << 20;
        long residentBytes = 0;
        bool hasEmbedding = false;
        foreach (var tensor in tensors)
        {
            if (tensor == null || tensor.Shape == null || tensor.Shape.Length == 0)
                throw new NotSupportedException("Streaming weight metadata must declare a non-empty tensor shape.");
            long elements = 1;
            foreach (ulong dimension in tensor.Shape)
            {
                if (dimension == 0 || dimension > long.MaxValue)
                    throw new NotSupportedException($"Streaming tensor '{tensor.Name}' has an invalid dimension.");
                elements = checked(elements * (long)dimension);
            }

            bool smallParameter = IsStreamingResidentParameter(tensor.Name);
            if (!smallParameter && tensor.Type == GgmlTensorType.Q8_0 && tensor.Shape.Length == 2
                && tensor.Shape[0] % 32 == 0 && IsStreamingProjection(tensor.Name))
            {
                hasEmbedding |= tensor.Name == "token_embd.weight";
                continue;
            }
            if (!smallParameter || tensor.Type != GgmlTensorType.F32)
                throw new NotSupportedException($"Streaming Qwen35 tensor '{tensor.Name}' ({tensor.Type}) is unsupported. " +
                    "Projection and embedding matrices must use Q8_0; only named F32 normalization/recurrent parameters and scalar scales remain resident.");
            if (tensor.Name.EndsWith(".scale", StringComparison.Ordinal) && elements != 1)
                throw new NotSupportedException($"Streaming sidecar scale '{tensor.Name}' must be scalar.");
            long bytes = checked(elements * sizeof(float));
            if (bytes > maximumTensorBytes || residentBytes > maximumResidentBytes - bytes)
                throw new NotSupportedException("Streaming Qwen35 small resident parameters exceed the 1 MiB per-tensor or 32 MiB total limit.");
            residentBytes += bytes;
        }
        if (!hasEmbedding)
            throw new NotSupportedException("Streaming Qwen35 requires a Q8_0 token_embd.weight table.");
        return residentBytes;
    }

    private static bool IsStreamingResidentParameter(string name)
    {
        if (name == "output_norm.weight") return true;
        if (name == null) return false;
        if (name.EndsWith(".scale", StringComparison.Ordinal))
            return name.StartsWith("blk.", StringComparison.Ordinal) || name is "output.scale" or "token_embd.scale";
        if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
        return name.EndsWith(".attn_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".post_attention_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".attn_q_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".attn_k_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".ssm_conv1d.weight", StringComparison.Ordinal)
            || name.EndsWith(".ssm_dt.bias", StringComparison.Ordinal)
            || name.EndsWith(".ssm_a", StringComparison.Ordinal)
            || name.EndsWith(".ssm_norm.weight", StringComparison.Ordinal);
    }

    private static bool IsStreamingProjection(string name)
        => name is "token_embd.weight" or "output.weight"
            || (name?.StartsWith("blk.", StringComparison.Ordinal) == true
                && (name.EndsWith(".attn_q.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_k.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_v.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_gate.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_output.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_up.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_down.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_gate_up.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ssm_in_proj.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ssm_alpha.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ssm_beta.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ssm_out.weight", StringComparison.Ordinal)));
}
