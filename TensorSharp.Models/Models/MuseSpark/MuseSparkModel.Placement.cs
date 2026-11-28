// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private int[] _layerRanks;

    private void BuildLayerPlacement()
    {
        // Plan from headers before LoadWeights can prefault/read payloads. A
        // stacked expert tensor is one resident allocation, not one allocation
        // plus another for each of its per-expert views. Fused QKVG replaces its
        // source matrices with an equal-sized allocation.
        var bytes = CapacityLayerBytes(MoeOffload.AllLayers
            ? Config.NumLayers : MoeOffload.CpuMoeLayers, out long embedding, out long head);
        _layerRanks = PlanLayerPlacement(bytes, embedding, head, LayerSplitDegree,
            Environment.GetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT"));
        if (LayerSplitDegree <= 1) return;
        var totals = new long[LayerSplitDegree];
        var counts = new int[LayerSplitDegree];
        totals[0] = embedding; totals[^1] += head;
        for (int l = 0; l < bytes.Length; l++) { totals[_layerRanks[l]] += bytes[l]; counts[_layerRanks[l]]++; }
        var parts = new List<string>();
        for (int rank = 0; rank < totals.Length; rank++)
            parts.Add($"gpu{rank}={counts[rank]} layers/{totals[rank] >> 20} MiB weights");
        Console.WriteLine("Muse Spark layer placement: " + string.Join(", ", parts) +
            "; KV, vision and compute scratch need additional memory.");
    }

    internal static long PlacementWeightBytes(GgufTensorInfo info, long rawBytes, BackendType backend)
        => ShouldStoreWeightQuantized(backend, info) ? rawBytes : checked(info.NumElements * sizeof(float));

    // Exact contiguous minimax partition (O(devices*layers^2), only 62 layers).
    // Roots stay on the first/last device. All requested devices own layers.
    internal static int[] PlanLayerPlacement(long[] layers, long embedding, long head, int devices, string counts = null)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (devices < 1 || devices > layers.Length || embedding < 0 || head < 0)
            throw new ArgumentException("Layer placement requires positive devices, at least one layer per device and nonnegative bytes.");
        var prefix = new long[layers.Length + 1];
        for (int l = 0; l < layers.Length; l++)
        {
            if (layers[l] < 0) throw new ArgumentOutOfRangeException(nameof(layers));
            prefix[l+1] = checked(prefix[l] + layers[l]);
        }
        var map = new int[layers.Length];
        if (!string.IsNullOrWhiteSpace(counts))
        {
            var parts = counts.Split(',');
            if (parts.Length != devices) throw new ArgumentException("TS_MUSE_SPARK_LAYER_SPLIT needs one positive count per GPU.");
            int cursor = 0;
            for (int rank = 0; rank < devices; rank++)
            {
                if (!int.TryParse(parts[rank], out int count) || count < 1 || count > layers.Length - cursor)
                    throw new ArgumentException("Invalid TS_MUSE_SPARK_LAYER_SPLIT layer count.");
                Array.Fill(map, rank, cursor, count);
                cursor += count;
            }
            if (cursor != layers.Length) throw new ArgumentException("TS_MUSE_SPARK_LAYER_SPLIT counts must sum to the model's layer count.");
            return map;
        }
        var best = new long[devices + 1, layers.Length + 1];
        var split = new int[devices + 1, layers.Length + 1];
        for (int d = 0; d <= devices; d++) for (int n = 0; n <= layers.Length; n++) best[d,n] = long.MaxValue;
        best[0,0] = 0;
        for (int d = 1; d <= devices; d++)
        for (int n = d; n <= layers.Length; n++)
        for (int start = d - 1; start < n; start++)
        {
            if (best[d-1,start] == long.MaxValue) continue;
            long own = checked(prefix[n] - prefix[start] + (d == 1 ? embedding : 0) + (d == devices ? head : 0));
            long cost = Math.Max(best[d-1,start], own);
            if (cost < best[d,n]) { best[d,n] = cost; split[d,n] = start; }
        }
        int end = layers.Length;
        for (int d = devices; d > 0; d--)
        {
            int start = split[d,end];
            Array.Fill(map, d-1, start, end-start);
            end = start;
        }
        return map;
    }
}
