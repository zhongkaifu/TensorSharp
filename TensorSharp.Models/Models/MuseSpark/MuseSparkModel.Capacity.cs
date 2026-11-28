// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    internal const string ScratchReserveVariable = "TS_MUSE_SPARK_SCRATCH_RESERVE_MIB";
    private const long MiB = 1L << 20;

    // Called after descriptor validation, before LoadWeights/prefault/preload.
    // TP needs its own sharded-weight accounting; whole-layer totals do not
    // describe a tensor-parallel rank.
    private void PreflightLayerSplitCapacity()
    {
        if (IsTensorParallel) return;
        BuildLayerPlacement();
        if (_backend is not (BackendType.GgmlCuda or BackendType.GgmlVulkan)) return;

        var totals = CapacityWeightTotals(MoeOffload.AllLayers
            ? Config.NumLayers : MoeOffload.CpuMoeLayers, _layerRanks);
        long[] free = QueryCapacityFreeBytes(LayerSplitDegree);
        long reserve = ResolveScratchReserveBytes(Config.HiddenSize, _latentSize,
            _expertFfnLength, _prefillChunk, Environment.GetEnvironmentVariable(ScratchReserveVariable));
        string reason = CapacityRefusal(totals, free, reserve);
        if (reason != null)
        {
            int? suggested = null;
            // Advice only: never change the requested placement/offload policy.
            // Only suggest a count when every selected device was measured.
            if (free.All(bytes => bytes >= 0))
            {
                int current = MoeOffload.AllLayers ? Config.NumLayers : MoeOffload.CpuMoeLayers;
                for (int count = current + 1; count <= Config.NumLayers; count++)
                {
                    var layers = CapacityLayerBytes(count, out long embedding, out long head);
                    int[] ranks = PlanLayerPlacement(layers, embedding, head, LayerSplitDegree,
                        Environment.GetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT"));
                    if (CapacityRefusal(SumPlacementBytes(layers, embedding, head, ranks, LayerSplitDegree), free, reserve) == null)
                    { suggested = count; break; }
                }
            }
            string advice = suggested.HasValue
                ? $"At least --n-cpu-moe {suggested.Value} fits this weight/headroom estimate with the selected layer split; verify host RAM and actual runtime memory."
                : "Increase --n-cpu-moe if host RAM permits, select more/larger GPUs, or revise TS_MUSE_SPARK_LAYER_SPLIT.";
            throw new ModelLoadRefusedException(reason + " " + advice +
                " Set an explicit MAX_CONTEXT and TS_KV_INITIAL_TOKENS (for example 16384 and 2048)." +
                " Reducing context alone cannot fix weights exceeding VRAM." +
                $" {ScratchReserveVariable} controls the estimated compute/KV reserve, not the weight requirement; " +
                "lower it only after measuring the chosen workload. No settings were changed.");
        }

        int context = ResolveConfiguredContextLength();
        int initial = ResolveInitialCacheAllocationLength(context, 2048);
        // GGML linear caches are host-backed; CUDA sessions retain active
        // working prefixes, not an entire maximum-length device cache per layer.
        // Do not falsely charge a lazily allocated 1M context to each GPU.
        bool compressed = _gguf.Tensors.Values.Any(IsQuantizedLinearWeight);
        int kvBytes = KvCacheDtypeConfig.IsExplicitlySet
            ? (KvCacheDtypeConfig.Current == KvCacheDtype.F32 ? 4 : 2)
            : (compressed ? 2 : _kvCacheDtype.ToDType().Size());
        long perToken = checked((long)Config.NumLayers * Config.NumKVHeads * _headDim * 2 * kvBytes);
        Console.WriteLine($"Muse Spark capacity: GPU weight bytes checked before payload reads; " +
            $"{reserve / MiB} MiB per GPU reserved as a compute/active-KV estimate. " +
            $"Host linear KV: {checked(perToken * initial) / MiB} MiB initially ({initial} tokens), " +
            $"{checked(perToken * context) / MiB} MiB at the {context}-token limit. " +
            "Vision, retained request caches and longer/batched workloads need additional memory; this is not a fit guarantee.");
        if (free.Any(bytes => bytes < 0))
            Console.WriteLine("Muse Spark capacity: device memory was unavailable on one or more ranks; those ranks were not checked.");
    }

    private long[] CapacityWeightTotals(int cpuLayers, int[] ranks)
    {
        var layers = CapacityLayerBytes(cpuLayers, out long embedding, out long head);
        return SumPlacementBytes(layers, embedding, head, ranks, LayerSplitDegree);
    }

    private long[] CapacityLayerBytes(int cpuLayers, out long embedding, out long head)
    {
        var layers = new long[Config.NumLayers];
        embedding = head = 0;
        foreach (var info in _gguf.Tensors.Values)
        {
            if (MoeCpuOffloadConfig.TryParseLayerIndex(info.Name, out int layer))
            {
                if ((uint)layer >= (uint)layers.Length)
                    throw new ArgumentException($"Muse Spark tensor {info.Name} names a layer outside the model.");
                bool expert = info.Name.Contains("_exps.", StringComparison.Ordinal);
                // F32 expert tensors use the ordinary tensor-op fallback,
                // so only preserved stacks qualify for native CPU offload.
                if (expert && layer < cpuLayers && IsQuantizedLinearWeight(info)) continue;
                layers[layer] = checked(layers[layer] + PlacementWeightBytes(info, _gguf.GetTensorByteCount(info), _backend));
            }
            else if (info.Name.StartsWith("output", StringComparison.Ordinal))
                head = checked(head + PlacementWeightBytes(info, _gguf.GetTensorByteCount(info), _backend));
            else
                embedding = checked(embedding + PlacementWeightBytes(info, _gguf.GetTensorByteCount(info), _backend));
        }
        return layers;
    }

    internal static long[] SumPlacementBytes(long[] layers, long embedding, long head, int[] ranks, int devices)
    {
        var totals = new long[devices];
        totals[0] = embedding;
        totals[^1] = checked(totals[^1] + head);
        for (int layer = 0; layer < layers.Length; layer++)
            totals[ranks[layer]] = checked(totals[ranks[layer]] + layers[layer]);
        return totals;
    }

    private static long[] QueryCapacityFreeBytes(int devices)
    {
        var result = new long[devices];
        Array.Fill(result, -1L);
        int previousRank = GgmlBasicOps.GetActiveRank();
        bool previousPin = GgmlBasicOps.PinRank(previousRank);
        try
        {
            for (int rank = 0; rank < devices; rank++)
            {
                GgmlBasicOps.SetActiveRank(rank);
                if (GgmlBasicOps.TryGetDeviceMemoryInfo(out long available, out long total) &&
                    available >= 0 && total > 0 && available <= total)
                    result[rank] = available;
            }
        }
        finally { GgmlBasicOps.RestoreRankPin(previousPin, previousRank); }
        return result;
    }

    internal static long ResolveScratchReserveBytes(int hidden, int latent, int ffn, int rows, string configuredMiB)
    {
        if (!string.IsNullOrWhiteSpace(configuredMiB))
        {
            if (!long.TryParse(configuredMiB, NumberStyles.None, CultureInfo.InvariantCulture, out long value) ||
                value < 0 || value > long.MaxValue / MiB)
                throw new ModelLoadRefusedException($"{ScratchReserveVariable} must be a nonnegative integer in MiB.");
            return value * MiB;
        }
        if (hidden <= 0 || latent <= 0 || ffn <= 0 || rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(hidden));
        // Planning headroom, not a claimed exact gallocr size. The published
        // 8192/4096/4096 model at 256 rows has required a >2.3 GiB allocation.
        // 2.5 GiB also preserves the measured five-A40/CPU32 F16 placement
        // (41.844 GiB weights on the fullest ranks, 44.424 GiB initially free).
        // Small checkpoints do not inherit an unconditional 2.5 GiB reservation.
        double scale = Math.Max(hidden / 8192.0, Math.Max(latent / 4096.0, ffn / 4096.0));
        double bytes = 2.5 * 1024.0 * MiB * scale * Math.Max(1, rows / 256.0);
        return checked((long)Math.Ceiling(Math.Max(16 * MiB, bytes)));
    }

    internal static string CapacityRefusal(long[] weights, long[] free, long reserve)
    {
        if (weights.Length != free.Length || reserve < 0) throw new ArgumentException("Invalid capacity inputs.");
        var failures = new List<string>();
        for (int rank = 0; rank < weights.Length; rank++)
        {
            if (weights[rank] < 0 || free[rank] < -1) throw new ArgumentException("Invalid capacity bytes.");
            if (free[rank] < 0) continue; // Unknown is not zero.
            if (weights[rank] > free[rank])
                failures.Add($"GPU rank {rank}: weights {weights[rank] / MiB} MiB exceed free VRAM {free[rank] / MiB} MiB");
            else if (reserve > free[rank] - weights[rank])
                failures.Add($"GPU rank {rank}: weights {weights[rank] / MiB} MiB plus estimated compute/active-KV headroom " +
                    $"{reserve / MiB} MiB exceed free VRAM {free[rank] / MiB} MiB");
        }
        return failures.Count == 0 ? null : "Muse Spark cannot fit the requested GPU placement: " + string.Join("; ", failures) + ".";
    }
}
