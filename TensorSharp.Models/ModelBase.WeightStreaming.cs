// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public abstract partial class ModelBase
{
    protected readonly WeightStreamingOptions WeightStreaming;
    private WeightStreamingExecutor _weightStreamingExecutor;
    private bool _streamingForwardFailed;
    protected bool HasStreamingWeights => WeightStreaming != null;
    // Match each model's ordinary projection policy: dense Qwen deliberately
    // keeps F32 activations, while Gemma uses the resident CUDA dispatch rules.
    protected virtual GgmlWeightStreamingArithmetic StreamingWeightArithmetic
        => GgmlWeightStreamingArithmetic.FullPrecision;

    /// <summary>Available only when explicit file-backed weight execution is enabled.
    /// These counters exclude ordinary model activations, KV and native graph caches.</summary>
    public WeightStreamingStatistics? StreamingWeightUsage => _weightStreamingExecutor?.Statistics;

    /// <summary>Read only the requested rows of any file-backed embedding table,
    /// including per-layer embeddings. The caller owns and releases the output.</summary>
    protected unsafe void ExecuteStreamedEmbedding(Tensor result, QuantizedWeight weight, int[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (_weightStreamingExecutor == null) throw new InvalidOperationException("The weight streaming owner has been disposed.");
        if (!weight.IsStreamed || result.DimensionCount != 2 || result.Sizes[0] != tokens.Length ||
            result.Sizes[1] != weight.Ne0 || result.ElementType != DType.Float32 || !result.IsContiguous())
            throw new ArgumentException("Streamed embedding requires a compatible contiguous Float32 output matrix.");
        _weightStreamingExecutor.Embedding(weight, tokens, (IntPtr)GetFloatPtr(result));
        InvalidateTensorDeviceCache(result);
    }

    private void ThrowIfStreamingStateFailed()
    {
        if (_streamingForwardFailed)
            throw new InvalidOperationException("A streamed forward failed after model state may have changed. " +
                "ResetKVCache and replay the request before forwarding again, or dispose the model.");
    }

    private void ReleaseStreamingWorkspaceForReset()
        => _weightStreamingExecutor?.ResetForReplay();

    protected void ExecuteQuantizedLinear(Tensor result, Tensor input, QuantizedWeight weight)
    {
        if (weight.IsStreamed)
        {
            if (_weightStreamingExecutor == null) throw new InvalidOperationException("The weight streaming owner has been disposed.");
            if (input.DimensionCount != 2 || result.DimensionCount != 2 || input.Sizes[1] != weight.Ne0 ||
                result.Sizes[1] != weight.Ne1 || input.Sizes[0] != result.Sizes[0] ||
                input.ElementType != DType.Float32 || result.ElementType != DType.Float32 || !result.IsContiguous())
                throw new ArgumentException("Streamed linear requires compatible Float32 matrices and contiguous output.");
            using Tensor contiguous = input.IsContiguous() ? null : Ops.NewContiguous(input);
            unsafe
            {
                _weightStreamingExecutor.Linear(weight, (IntPtr)GetFloatPtr(contiguous ?? input),
                    (IntPtr)GetFloatPtr(result), checked((int)input.Sizes[0]));
            }
            InvalidateTensorDeviceCache(result);
        }
        else if (IsGgmlBackend)
            GgmlBasicOps.AddmmQuant(result, input, weight.CacheKey, weight.GgmlType, weight.Ne0, weight.Ne1, weight.RawBytes);
        else
            AddmmQuantManaged(result, input, weight);
    }
}
