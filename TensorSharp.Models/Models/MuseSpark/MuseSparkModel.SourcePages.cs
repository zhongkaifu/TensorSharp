// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _releaseGpuSourcePages =
        Environment.GetEnvironmentVariable("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES") == "1";
    private readonly HashSet<string> _advisedGpuSourceWeights = new(StringComparer.Ordinal);
    private readonly HashSet<int> _advisedGpuExpertLayers = new();
    private static readonly string[] ExpertSourceSuffixes =
        ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
    internal long AdvisedGpuSourceBytes { get; private set; }
    internal int AdvisedGpuSourceWeightCount => _advisedGpuSourceWeights.Count;

    internal static bool CanReleaseGpuSourcePages(bool enabled, BackendType backend, bool offloaded)
        => enabled && backend == BackendType.GgmlCuda && !offloaded;

    private void ReleasePreloadedGpuSourcePages()
    {
        if (!CanReleaseGpuSourcePages(_releaseGpuSourcePages, _backend, false) || !OperatingSystem.IsLinux()) return;
        int previousRank = GgmlBasicOps.GetActiveRank();
        try
        {
            foreach (var (name, weight) in _quantWeights)
            {
                // Embedding rows may still be gathered on the CPU. Fused/copied
                // projections with no original GGUF tensor are also excluded.
                if (name == "token_embd.weight" || !weight.IsFileBacked || weight.DevicePreloadTooLarge ||
                    !ShouldPreloadCudaQuantWeightToDevice(name)) continue;
                GgmlBasicOps.SetActiveRank(PreloadRankForWeight(name));
                TryReleaseGpuSourcePages(name, weight.Data, weight.RawBytes, weight.CacheKey);
            }
        }
        finally { GgmlBasicOps.SetActiveRank(previousRank); }
        LogGpuSourcePageAdvice("preloaded weights");
    }

    private void ReleaseExpertGpuSourcePages(int layer)
    {
        // CPU experts include the optional streamed-to-GPU prefill route. They
        // must remain host-resident for the next decode or streamed prefill.
        if (!CanReleaseGpuSourcePages(_releaseGpuSourcePages, _backend, MoeOffload.IsLayerOnCpu(layer)) ||
            !OperatingSystem.IsLinux() || _advisedGpuExpertLayers.Contains(layer)) return;
        int before = _advisedGpuSourceWeights.Count;
        bool complete = true;
        foreach (string suffix in ExpertSourceSuffixes)
        {
            string name = _prefixes[layer] + suffix;
            if (_stackedExpertWeights.TryGetValue(name, out var weight) && weight.IsExternalView)
                TryReleaseGpuSourcePages(name, weight.Data, weight.TotalRawBytes, weight.Data);
            complete &= _advisedGpuSourceWeights.Contains(name);
        }
        if (complete) _advisedGpuExpertLayers.Add(layer);
        if (_advisedGpuSourceWeights.Count != before) LogGpuSourcePageAdvice($"layer {layer} experts");
    }

    // Runs on the caller after every rank has completed its preload. Never
    // mutate the advice sets from a rank worker. A resident slice alone does
    // not justify dropping the complete tensor's mapped source pages.
    private void ReleaseTensorParallelGpuSourcePages()
    {
        if (!IsTensorParallel || !CanReleaseGpuSourcePages(_releaseGpuSourcePages, _backend, false) ||
            !OperatingSystem.IsLinux()) return;
        int previousRank = GgmlBasicOps.GetActiveRank();
        bool previousPin = GgmlBasicOps.PinRank(previousRank);
        try
        {
            foreach (var (name, shards) in _tpQuantWeights)
            {
                if (!_gguf.Tensors.TryGetValue(name, out var info) || shards.Length != TpDegree) continue;
                bool resident = true;
                long bytes = 0;
                for (int rank = 0; rank < TpDegree; rank++)
                {
                    var shard = shards[rank];
                    if (shard == null || shard.DevicePreloadTooLarge) { resident = false; break; }
                    GgmlBasicOps.SetActiveRank(rank);
                    if (!GgmlBasicOps.HasResidentDeviceCopy(shard.CacheKey, shard.RawBytes))
                    { resident = false; break; }
                    bytes = checked(bytes + shard.RawBytes);
                }
                if (resident && bytes == _gguf.GetTensorByteCount(info) &&
                    _gguf.TryGetTensorDataPointer(info, out IntPtr mapped))
                    AdviseResidentTensorSource(name, info, mapped, bytes);
            }
            if (_sparkTpGate == null) return;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (MoeOffload.IsLayerOnCpu(layer) || _sparkTpGate[layer] == null) continue;
                var collections = new[] { _sparkTpGate[layer], _sparkTpUp[layer], _sparkTpDown[layer] };
                for (int projection = 0; projection < collections.Length; projection++)
                {
                    string name = _prefixes[layer] + ExpertSourceSuffixes[projection];
                    if (!_stackedExpertWeights.TryGetValue(name, out var source) || !source.IsExternalView ||
                        !_gguf.Tensors.TryGetValue(name, out var info)) continue;
                    long bytes = 0;
                    bool resident = true;
                    for (int rank = 0; rank < TpDegree; rank++)
                    {
                        var shard = collections[projection][rank];
                        // The intervals must cover the original stacked tensor
                        // exactly, with neither gaps nor duplicate expert ranges.
                        if (shard == null || shard.Data != source.Data + (nint)bytes)
                        { resident = false; break; }
                        GgmlBasicOps.SetActiveRank(rank);
                        if (!GgmlBasicOps.HasResidentDeviceCopy(shard.Data, shard.TotalRawBytes))
                        { resident = false; break; }
                        bytes = checked(bytes + shard.TotalRawBytes);
                    }
                    if (resident && bytes == source.TotalRawBytes)
                        AdviseResidentTensorSource(name, info, source.Data, bytes);
                }
            }
        }
        finally { GgmlBasicOps.RestoreRankPin(previousPin, previousRank); }
        LogGpuSourcePageAdvice("tensor-parallel shards");
    }

    private void TryReleaseGpuSourcePages(string name, IntPtr data, long bytes, IntPtr cacheKey)
    {
        if (_advisedGpuSourceWeights.Contains(name) || !_gguf.Tensors.TryGetValue(name, out var info) ||
            !GgmlBasicOps.HasResidentDeviceCopy(cacheKey, bytes)) return;
        AdviseResidentTensorSource(name, info, data, bytes);
    }

    private void AdviseResidentTensorSource(string name, GgufTensorInfo info, IntPtr data, long bytes)
    {
        if (_advisedGpuSourceWeights.Contains(name)) return;
        // The GGUF owner verifies pointer identity and full tensor length before
        // advising anything; an external view alone is not proof of a file map.
        if (!_gguf.TryReleaseMappedTensorPages(info, data, bytes, out long advisedBytes)) return;
        _advisedGpuSourceWeights.Add(name);
        AdvisedGpuSourceBytes += advisedBytes;
    }

    private void LogGpuSourcePageAdvice(string stage)
        => Console.WriteLine($"Muse Spark GPU source-page advice ({stage}): " +
            $"{AdvisedGpuSourceWeightCount} weights, {AdvisedGpuSourceBytes} bytes " +
            $"({AdvisedGpuSourceBytes / 1073741824.0:F2} GiB) cumulatively advised; mappings retained.");
}
