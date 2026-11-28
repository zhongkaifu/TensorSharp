// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private const float RouteWeightSumFloor = 6.103515625e-5f;
    private readonly Dictionary<int, float[]> _expertPostNorm = new();

    /// <summary>Bias chooses the experts but does not enter their output weights.</summary>
    internal static void Route(ReadOnlySpan<float> logits, ReadOnlySpan<float> bias, int used,
        bool normalize, float scale, Span<int> selected, Span<float> weights)
    {
        if (used <= 0 || used > logits.Length || bias.Length != logits.Length ||
            selected.Length < used || weights.Length < used)
            throw new ArgumentException("Invalid Muse Spark router dimensions.");
        Span<float> scores = logits.Length <= 1024 ? stackalloc float[logits.Length] : new float[logits.Length];
        for (int e = 0; e < logits.Length; e++)
        {
            if (!float.IsFinite(logits[e]) || !float.IsFinite(bias[e]))
                throw new InvalidOperationException("Muse Spark router received a non-finite logit or bias.");
            scores[e] = TensorComputePrimitives.Sigmoid(logits[e]) + bias[e];
        }
        float sum = 0;
        for (int k = 0; k < used; k++)
        {
            int best = 0;
            for (int e = 1; e < scores.Length; e++) if (scores[e] > scores[best]) best = e;
            selected[k] = best;
            scores[best] = float.NegativeInfinity;
            weights[k] = TensorComputePrimitives.Sigmoid(logits[best]);
            sum += weights[k];
        }
        float factor = normalize ? 1f / MathF.Max(sum, RouteWeightSumFloor) : 1f;
        // GGUF zero denotes unspecified scale, matching build_moe_ffn.
        if (scale != 0f) factor *= scale;
        for (int k = 0; k < used; k++) weights[k] *= factor;
    }

    private Tensor MoE(Tensor input, int layer, int rows)
    {
        _sparkPhaseFrame?.Next(MuseSparkPhase.Router);
        string p = _prefixes[layer];
        using var logits = LinearForward(input, p + "ffn_gate_inp.weight");
        float[] router = TensorToFloatArray(logits);
        float[] bias = TensorToFloatArray(_weights[p + "exp_probs_b.bias"]);
        var selected = new int[rows * _numExpertsUsed];
        var routing = new float[selected.Length];
        for (int row = 0; row < rows; row++)
            Route(router.AsSpan(row * _numExperts, _numExperts), bias, _numExpertsUsed,
                _expertWeightsNorm, _expertScale,
                selected.AsSpan(row * _numExpertsUsed, _numExpertsUsed),
                routing.AsSpan(row * _numExpertsUsed, _numExpertsUsed));
        _sparkPhaseFrame?.Next(MuseSparkPhase.LatentPreparation);
        using var latent = LinearForward(input, p + "ffn_latent_down.weight");
        using var expertInput = Norm(latent, Config.Eps, p + "ffn_pre_exp_norm.weight");
        if (IsTensorParallel && !MoeOffload.IsLayerOnCpu(layer))
        {
            using var parallelOutput = TensorParallelExperts(expertInput, layer, rows, selected, routing);
            _sparkPhaseFrame?.Next(MuseSparkPhase.LatentUp);
            return LinearForward(parallelOutput, p + "ffn_latent_up.weight");
        }
        if (_sparkPhaseFrame != null)
        {
            bool nativeCpu = IsGgmlBackend && MoeOffload.IsLayerOnCpu(layer) &&
                _stackedExpertWeights.ContainsKey(p + "ffn_gate_exps.weight") &&
                _stackedExpertWeights.ContainsKey(p + "ffn_up_exps.weight") &&
                _stackedExpertWeights.ContainsKey(p + "ffn_down_exps.weight");
            _sparkPhaseFrame.Next(nativeCpu ? MuseSparkPhase.CpuExperts : MuseSparkPhase.ExpertsSingle);
        }
        using var expertOutput = new Tensor(_allocator, DType.Float32, rows, _latentSize);
        if (!TryNativeExperts(expertInput, expertOutput, layer, rows, selected, routing))
            GroupedExperts(expertInput, expertOutput, layer, rows, selected, routing);
        _sparkPhaseFrame?.Next(MuseSparkPhase.LatentUp);
        return LinearForward(expertOutput, p + "ffn_latent_up.weight");
    }

    private bool TryNativeExperts(Tensor input, Tensor output, int layer, int rows, int[] selected, float[] routing)
    {
        if (!IsGgmlBackend) return false;
        string p = _prefixes[layer];
        if (!_stackedExpertWeights.TryGetValue(p + "ffn_gate_exps.weight", out var gate) ||
            !_stackedExpertWeights.TryGetValue(p + "ffn_up_exps.weight", out var up) ||
            !_stackedExpertWeights.TryGetValue(p + "ffn_down_exps.weight", out var down)) return false;
        if (!_expertPostNorm.TryGetValue(layer, out var norm))
            _expertPostNorm[layer] = norm = TensorToFloatArray(_weights[p + "ffn_post_exp_norm.weight"]);
        GgmlBasicOps.MoEFFNPrefill(input, output, rows, _latentSize, _expertFfnLength,
            _numExperts, _numExpertsUsed, selected, routing,
            gate.Data, gate.GgmlType, gate.PerExpertNe0, gate.PerExpertNe1, gate.TotalRawBytes,
            up.Data, up.GgmlType, up.PerExpertNe0, up.PerExpertNe1, up.TotalRawBytes,
            down.Data, down.GgmlType, down.PerExpertNe0, down.PerExpertNe1, down.TotalRawBytes,
            null, null, null, GgmlBasicOps.MoEActivation.SwiGLUSplit, runOnCpu: MoeOffload.IsLayerOnCpu(layer),
            postExpertNorm: norm, postExpertNormEps: _postNormEps);
        ReleaseExpertGpuSourcePages(layer);
        InvalidateTensorDeviceCache(output);
        return true;
    }

    /// <summary>Bucket token routes by expert to batch all rows through three GEMMs.
    /// Crucially, scale then RMS-normalize each expert result before the mixture.</summary>
    private unsafe void GroupedExperts(Tensor input, Tensor output, int layer, int rows, int[] selected, float[] routing)
    {
        var buckets = new List<int>[_numExperts];
        for (int route = 0; route < selected.Length; route++)
            (buckets[selected[route]] ??= new List<int>()).Add(route);
        float* source = GetFloatPtr(input), result = GetFloatPtr(output);
        new Span<float>(result, rows * _latentSize).Clear();
        for (int expert = 0; expert < _numExperts; expert++)
        {
            var routes = buckets[expert];
            if (routes == null) continue;
            using var gathered = new Tensor(_allocator, DType.Float32, routes.Count, _latentSize);
            float* gp = GetFloatPtr(gathered);
            for (int r = 0; r < routes.Count; r++)
                new ReadOnlySpan<float>(source + (long)(routes[r] / _numExpertsUsed) * _latentSize, _latentSize)
                    .CopyTo(new Span<float>(gp + (long)r * _latentSize, _latentSize));
            InvalidateTensorDeviceCache(gathered);
            using var gate = ExpertLinear(gathered, layer, expert, "gate", _expertFfnLength);
            using var up = ExpertLinear(gathered, layer, expert, "up", _expertFfnLength);
            Ops.SiLUMul(gate, gate, up);
            using var down = ExpertLinear(gate, layer, expert, "down", _latentSize);
            using var scaled = ScaleChannels(down, _prefixes[layer] + "ffn_post_exp_norm.weight");
            using var normalized = Norm(scaled, _postNormEps);
            float* np = GetFloatPtr(normalized);
            for (int r = 0; r < routes.Count; r++)
            {
                int route = routes[r], token = route / _numExpertsUsed;
                TensorComputePrimitives.ScaleAdd(result + (long)token * _latentSize,
                    np + (long)r * _latentSize, routing[route], _latentSize);
            }
        }
        InvalidateTensorDeviceCache(output);
    }

    private Tensor ExpertLinear(Tensor input, int layer, int expert, string kind, int width)
    {
        string p = _prefixes[layer] + "ffn_" + kind + "_exps";
        if (_quantWeights.TryGetValue(p + $".{expert}.weight", out var weight))
        {
            var result = new Tensor(_allocator, DType.Float32, input.Sizes[0], width);
            if (IsGgmlBackend && !MoeOffload.IsLayerOnCpu(layer))
                GgmlBasicOps.AddmmQuant(result, input, weight.CacheKey, weight.GgmlType, weight.Ne0, weight.Ne1, weight.RawBytes);
            else
                AddmmQuantManaged(result, input, weight);
            if (weight.Scale != 1f) Ops.Mul(result, result, weight.Scale);
            return result;
        }
        using var plane = _weights[p + ".weight"].Select(0, expert);
        var output = new Tensor(_allocator, DType.Float32, input.Sizes[0], width);
        try
        {
            SparkFloatProjection(output, input, plane);
            return output;
        }
        catch { output.Dispose(); throw; }
    }
}
