// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class Gemma4Model
{
    protected override GgmlWeightStreamingArithmetic StreamingWeightArithmetic => GgmlWeightStreamingArithmetic.ResidentCuda;

    /// <summary>Small resident F32 parameters, excluding activations, live KV,
    /// backend pools and the separately budgeted file-weight workspaces.</summary>
    public long StreamingResidentParameterBytes { get; private set; }

    // Match the resident matmul's logical output-row geometry. CUDA MMQ's
    // stream-K reduction depends on that geometry, even when the input and
    // original quantized rows are identical. These are file-source views:
    // actual staging remains bounded and charged by the streaming executor.
    private void ComposeStreamingProjectionWeights()
    {
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            string prefix = $"blk.{layer}";
            if (!_kvDonorMap.ContainsKey(layer)
                && _quantWeights.TryGetValue($"{prefix}.attn_q.weight", out var q)
                && _quantWeights.TryGetValue($"{prefix}.attn_k.weight", out var k))
            {
                bool hasV = _quantWeights.TryGetValue($"{prefix}.attn_v.weight", out var v);
                v ??= k;
                if (q.Scale != 1f || k.Scale != 1f || v.Scale != 1f)
                    throw new NotSupportedException("Streaming Gemma4 QKV composition requires unit projection scales.");
                var composite = QuantizedWeight.CreateFileBackedConcatenation(q, k, v);
                ReplaceStreamingProjection($"{prefix}.attn_qkv.weight", composite,
                    hasV ? new[] { $"{prefix}.attn_q.weight", $"{prefix}.attn_k.weight", $"{prefix}.attn_v.weight" }
                         : new[] { $"{prefix}.attn_q.weight", $"{prefix}.attn_k.weight" });
            }
            if (_quantWeights.TryGetValue($"{prefix}.ffn_gate.weight", out var gate)
                && _quantWeights.TryGetValue($"{prefix}.ffn_up.weight", out var up)
                && gate.Scale == up.Scale)
            {
                var composite = QuantizedWeight.CreateFileBackedConcatenation(gate, up);
                ReplaceStreamingProjection($"{prefix}.ffn_gate_up.weight", composite,
                    new[] { $"{prefix}.ffn_gate.weight", $"{prefix}.ffn_up.weight" });
            }
        }
    }

    private void ReplaceStreamingProjection(string name, QuantizedWeight composite, string[] sourceNames)
    {
        // Publication precedes release, so constructor cleanup owns the new
        // descriptor even if an unexpected source cleanup fails.
        _quantWeights.TryGetValue(name, out var previous);
        _quantWeights[name] = composite;
        previous?.Dispose();
        foreach (string sourceName in sourceNames)
        {
            var source = _quantWeights[sourceName];
            source.Dispose();
            _quantWeights.Remove(sourceName);
        }
    }

    private void RefuseStreamingAlternateEntry(string operation)
    {
        if (HasStreamingWeights)
            throw new NotSupportedException($"File-backed Gemma4 weights support sequential text Forward/ForwardRefill only; {operation} is unsupported.");
    }

    private void ValidateStreamingWeightConfiguration(string draftModelPath)
    {
        if (!HasStreamingWeights) return;
        StreamingResidentParameterBytes = ValidateStreamingWeightMetadata(Config.Architecture, _backend,
            GlobalTpDegree, LayerSplitDegree, _numExperts,
            checked((int)_gguf.GetUint32($"{Config.Architecture}.nextn_predict_layers", 0)),
            draftModelPath, _gguf.Tensors.Values, _kvCacheDtype);
        Console.WriteLine($"  File-backed Gemma4 weights: original Q8_0 matrices and F16 PLE projection, " +
            $"synchronous row tiles; small resident F32 parameters: {StreamingResidentParameterBytes} bytes " +
            "(outside the weight staging budget; activations and KV are separate).");
    }

    internal static long ValidateStreamingWeightMetadata(string architecture, BackendType backend,
        int tensorParallelDegree, int layerSplitDegree, int experts, int mtpLayers,
        string draftModelPath, IEnumerable<GgufTensorInfo> tensors, KvCacheDtype kvCacheDtype = KvCacheDtype.F16)
    {
        if (architecture != "gemma4" || backend != BackendType.GgmlCuda
            || tensorParallelDegree != 1 || layerSplitDegree != 1
            || experts != 0 || mtpLayers != 0 || !string.IsNullOrEmpty(draftModelPath))
            throw new NotSupportedException("File-backed Gemma4 weights require dense gemma4, one GGML CUDA rank, " +
                "and no MTP or external draft model. Tensor parallelism, layer splitting and other backends are unsupported.");
        if (kvCacheDtype is not (KvCacheDtype.F16 or KvCacheDtype.F32))
            throw new NotSupportedException("File-backed Gemma4 weights require F16 or F32 KV storage; quantized KV is unsupported.");

        const long maximumTensorBytes = 1L << 20, maximumResidentBytes = 32L << 20;
        long residentBytes = 0;
        bool hasEmbedding = false;
        foreach (var tensor in tensors)
        {
            if (tensor?.Shape == null || tensor.Shape.Length == 0)
                throw new NotSupportedException("Streaming weight metadata must declare a non-empty tensor shape.");
            long elements = 1;
            foreach (ulong dimension in tensor.Shape)
            {
                if (dimension == 0 || dimension > int.MaxValue)
                    throw new NotSupportedException($"Streaming tensor '{tensor.Name}' has an invalid dimension.");
                elements = checked(elements * (long)dimension);
            }
            bool smallParameter = IsStreamingResidentParameter(tensor.Name);
            bool supportedMatrixType = tensor.Type == GgmlTensorType.Q8_0
                || (tensor.Type == GgmlTensorType.F16 && tensor.Name == "per_layer_model_proj.weight");
            if (tensor.Type == GgmlTensorType.F16 && tensor.Name == "per_layer_model_proj.weight"
                && (tensor.Shape.Length != 2 || tensor.Shape[0] % 64 != 0 || tensor.Shape[1] % 32 != 0))
                throw new NotSupportedException("Streaming Gemma4 F16 PLE projection requires input width divisible by 64 " +
                    "and output rows divisible by 32 for resident CUDA arithmetic.");
            if (!smallParameter && supportedMatrixType && tensor.Shape.Length == 2
                && (tensor.Type != GgmlTensorType.Q8_0 || tensor.Shape[0] % 32 == 0)
                && IsStreamingProjection(tensor.Name))
            {
                hasEmbedding |= tensor.Name == "token_embd.weight";
                continue;
            }
            if (!smallParameter || tensor.Type != GgmlTensorType.F32 || tensor.Shape.Length != 1)
                throw new NotSupportedException($"Streaming Gemma4 tensor '{tensor.Name}' ({tensor.Type}) is unsupported. " +
                    "Matrices must use Q8_0, except the explicitly streamed F16 per_layer_model_proj.weight. " +
                    "Only named small F32 vectors and scalar scales remain resident.");
            if ((tensor.Name.EndsWith(".scale", StringComparison.Ordinal)
                    || tensor.Name.EndsWith(".layer_output_scale.weight", StringComparison.Ordinal)) && elements != 1)
                throw new NotSupportedException($"Streaming scale '{tensor.Name}' must be scalar.");
            long bytes = checked(elements * sizeof(float));
            if (bytes > maximumTensorBytes || residentBytes > maximumResidentBytes - bytes)
                throw new NotSupportedException("Streaming Gemma4 small resident parameters exceed the 1 MiB per-tensor or 32 MiB total limit.");
            residentBytes += bytes;
        }
        if (!hasEmbedding)
            throw new NotSupportedException("Streaming Gemma4 requires a Q8_0 token_embd.weight table.");
        return residentBytes;
    }

    private static bool IsStreamingResidentParameter(string name)
    {
        if (name is "output_norm.weight" or "per_layer_proj_norm.weight" or "rope_freqs.weight") return true;
        if (name == null) return false;
        if (name.EndsWith(".scale", StringComparison.Ordinal))
            return name.StartsWith("blk.", StringComparison.Ordinal)
                || name is "output.scale" or "token_embd.scale" or "per_layer_model_proj.scale" or "per_layer_token_embd.scale";
        if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
        return name.EndsWith(".attn_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".post_attention_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".attn_q_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".attn_k_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".ffn_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".post_ffw_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".ffn_post_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".post_norm.weight", StringComparison.Ordinal)
            || name.EndsWith(".layer_output_scale.weight", StringComparison.Ordinal);
    }

    private static bool IsStreamingProjection(string name)
        => name is "token_embd.weight" or "output.weight" or "per_layer_token_embd.weight" or "per_layer_model_proj.weight"
            || (name?.StartsWith("blk.", StringComparison.Ordinal) == true
                && (name.EndsWith(".attn_q.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_k.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_v.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)
                    || name.EndsWith(".attn_output.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_up.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_down.weight", StringComparison.Ordinal)
                    || name.EndsWith(".ffn_gate_up.weight", StringComparison.Ordinal)
                    || name.EndsWith(".inp_gate.weight", StringComparison.Ordinal)
                    || name.EndsWith(".proj.weight", StringComparison.Ordinal)));

    // The resident whole-model graph attends values rounded to the cache's
    // dtype, including a fresh SWA chunk that is larger than the ring. Keep
    // that chunk intact here: reading it back from the ring would discard the
    // early keys still needed by early queries and downstream shared layers.
    private Tensor StreamingPrefillAttention(Tensor q, Tensor k, Tensor v,
        int kvHeads, int headDim, int seqLen, int kvLen, int window, DType cacheType)
    {
        var result = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * headDim);
        try
        {
            if (cacheType == DType.Float16)
            {
                using var roundedK = new Tensor(_allocator, DType.Float16, kvHeads, kvLen, headDim);
                using var roundedV = new Tensor(_allocator, DType.Float16, kvHeads, kvLen, headDim);
                CopyToCache(roundedK, k, 0, kvLen);
                CopyToCache(roundedV, v, 0, kvLen);
                GgmlBasicOps.StreamingFlashAttention(q, roundedK, roundedV, result,
                    Config.NumHeads, kvHeads, headDim, seqLen, kvLen, kvLen,
                    kvLen - seqLen, window, 1f);
            }
            else
                GgmlBasicOps.StreamingFlashAttention(q, k, v, result,
                    Config.NumHeads, kvHeads, headDim, seqLen, kvLen, kvLen,
                    kvLen - seqLen, window, 1f);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private Tensor StreamingCacheAttention(Tensor q, Tensor k, Tensor v,
        int kvHeads, int headDim, int seqLen, int kvLen)
    {
        var result = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * headDim);
        try
        {
            GgmlBasicOps.StreamingFlashAttention(q, k, v, result,
                Config.NumHeads, kvHeads, headDim, seqLen, kvLen, (int)k.Sizes[1],
                kvLen - seqLen, 0, 1f);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    // The PLE table is much larger than the normal embedding table on E4B.
    // Consume only requested rows; neither get_rows nor a graph may retain it.
    private Tensor ComputeStreamingPLE(int[] tokens, Tensor hiddenState, int seqLen)
    {
        Tensor tokenEmbedding = null, projection = null;
        try
        {
            int totalPleDim = checked(_pleDim * Config.NumLayers);
            if (_quantWeights.TryGetValue("per_layer_token_embd.weight", out var weight))
            {
                tokenEmbedding = new Tensor(_allocator, DType.Float32, seqLen, totalPleDim);
                ExecuteStreamedEmbedding(tokenEmbedding, weight, tokens);
                Ops.Mul(tokenEmbedding, tokenEmbedding, MathF.Sqrt(_pleDim));
                DumpStreamingTensor(tokenEmbedding, "pleTokenScaled");
            }
            projection = LinearForward(hiddenState, "per_layer_model_proj.weight");
            if (projection != null)
            {
                DumpStreamingTensor(projection, "pleProjectionRaw");
                Ops.Mul(projection, projection, 1f / MathF.Sqrt(Config.HiddenSize));
                DumpStreamingTensor(projection, "pleProjectionScaled");
                using var reshaped = projection.View(checked(seqLen * Config.NumLayers), _pleDim);
                Ops.RMSNorm(reshaped, reshaped, _weights["per_layer_proj_norm.weight"], null, Config.Eps);
                DumpStreamingTensor(projection, "pleProjectionNormed");
            }
            if (projection != null && tokenEmbedding != null)
            {
                Ops.Add(projection, projection, tokenEmbedding);
                Ops.Mul(projection, projection, 1f / MathF.Sqrt(2f));
            }
            Tensor result = projection ?? tokenEmbedding;
            DumpStreamingTensor(result, "pleCombined");
            if (ReferenceEquals(result, projection)) projection = null;
            else tokenEmbedding = null;
            return result;
        }
        finally
        {
            tokenEmbedding?.Dispose();
            projection?.Dispose();
        }
    }

    // Keep transient ownership explicit when any file read or shared-budget
    // acquisition can fail between projections. The public Forward wrapper
    // latches the failed state until ResetKVCache succeeds.
    private unsafe float[] ForwardStreamingCore(int[] tokens, bool produceLogits)
    {
        _forwardSw.Start();
        try
        {
            int seqLen = tokens.Length, startPos = _cacheSeqLen;
            BeginStreamingDiagnostic(tokens, startPos);
            EnsureCacheCapacity(checked(startPos + seqLen));
            EnsureKvCacheHostSynchronized();
            using var hidden = Embedding(tokens);
            ScaleEmbedding(hidden);
            DumpStreamingTensor(hidden, "embeddingScaled");
            using var perLayerInputs = _pleDim > 0 ? ComputeStreamingPLE(tokens, hidden, seqLen) : null;
            if (_swaKVDonorLayers.Count > 0 && seqLen > 1)
                _prefillSWAKV = new Dictionary<int, (Tensor, Tensor)>();
            if (seqLen > 1) PrepareSwaPrevWindowsForChunk(startPos, seqLen);
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _gemmaDiagnosticCurrentLayer = layer;
                using var perLayerInput = perLayerInputs == null ? null : ExtractPerLayerSlice(perLayerInputs, layer, seqLen);
                string prefix = $"blk.{layer}";
                using (var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight"))
                {
                    DumpStreamingTensor(normed, $"layer{layer:D2}.attnNorm", layer);
                    using var attention = Attention(normed, layer, prefix, seqLen, startPos, _kvDonorMap.ContainsKey(layer));
                    DumpStreamingTensor(attention, $"layer{layer:D2}.attentionProjected", layer);
                    GgmlBasicOps.StreamingNormResidual(hidden, attention, _weights[$"{prefix}.post_attention_norm.weight"], Config.Eps);
                }
                DumpStreamingTensor(hidden, $"layer{layer:D2}.attentionResidual");
                string postFfnNorm = _weights.ContainsKey($"{prefix}.post_ffw_norm.weight")
                    ? $"{prefix}.post_ffw_norm.weight" : $"{prefix}.ffn_post_norm.weight";
                using (var ffn = FFNGeluWithOptionalNorm(hidden, $"{prefix}.ffn_norm.weight",
                    $"{prefix}.ffn_gate_up.weight", $"{prefix}.ffn_down.weight", seqLen))
                {
                    GgmlBasicOps.StreamingNormResidual(hidden, ffn, _weights[postFfnNorm], Config.Eps);
                }
                DumpStreamingTensor(hidden, $"layer{layer:D2}.ffnResidual");
                if (perLayerInput != null && HasLinearWeight($"{prefix}.inp_gate.weight"))
                {
                    using var gate = LinearForward(hidden, $"{prefix}.inp_gate.weight");
                    Ops.GELUMul(gate, gate, perLayerInput);
                    using var projection = LinearForward(gate, $"{prefix}.proj.weight");
                    if (projection != null)
                        GgmlBasicOps.StreamingNormResidual(hidden, projection, _weights[$"{prefix}.post_norm.weight"], Config.Eps);
                }
                if (_layerScalars[layer] != 1f) Ops.Mul(hidden, hidden, _layerScalars[layer]);
                DumpStreamingTensor(hidden, $"layer{layer:D2}.output");
            }
            if (produceLogits)
            {
                using var lastRow = hidden.Narrow(0, seqLen - 1, 1);
                using var lastHidden = Ops.NewContiguous(lastRow);
                Ops.RMSNorm(lastHidden, lastHidden, _weights["output_norm.weight"], null, Config.Eps);
                long started = Stopwatch.GetTimestamp();
                using var logits = LinearForward(lastHidden, _hasTiedOutput ? "token_embd.weight" : "output.weight");
                _lmHeadTicks += Stopwatch.GetTimestamp() - started;
                if (_finalLogitSoftcap > 0f) ApplyLogitSoftcap(logits);
                _logitsBuffer ??= new float[Config.VocabSize];
                fixed (float* destination = _logitsBuffer)
                    Buffer.MemoryCopy(GetFloatPtr(logits), destination, checked(Config.VocabSize * 4L), checked(Config.VocabSize * 4L));
            }
            _cacheSeqLen += seqLen;
            _forwardCount++;
            return produceLogits ? _logitsBuffer : null;
        }
        finally
        {
            _gemmaDiagnosticTag = null;
            if (_prefillSWAKV != null)
            {
                foreach (var kv in _prefillSWAKV.Values) { kv.k.Dispose(); kv.v.Dispose(); }
                _prefillSWAKV = null;
            }
            DisposeSwaPrevWindows();
            _forwardSw.Stop();
        }
    }
}
