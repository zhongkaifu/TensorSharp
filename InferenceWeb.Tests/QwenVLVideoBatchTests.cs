// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="Qwen35VisionEncoder.EncodeVideoPairs"/> on the GGML CPU backend, where the merged tower graph stacks a clip's
/// temporal pairs (one graph, attention within each pair) and rotates Q and K with ggml's vision rope. Each stacked pair
/// must be the pair encoded on its own, and both must be the managed CPU tower's cos/sin-table result. Synthetic one-block
/// projector (<see cref="QwenVLSyntheticMmprojBuilder"/>): wiring and numerics, not encoder quality.
/// </summary>
public sealed class QwenVLVideoBatchTests : IDisposable
{
    private const int Side = 64;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "qwenvl-batch-" + Guid.NewGuid().ToString("N"));
    private readonly string _mmproj;

    public QwenVLVideoBatchTests()
    {
        Directory.CreateDirectory(_directory);
        _mmproj = QwenVLSyntheticMmprojBuilder.Write(Path.Combine(_directory, "mmproj.gguf"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void StackedPairsEqualPairsEncodedAloneAndTheManagedTower()
    {
        var firsts = new List<float[]> { Pixels(1), Pixels(3), Pixels(5) };
        var seconds = new List<float[]> { Pixels(2), Pixels(4), Pixels(6) };

        var ggml = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        using var encoder = new Qwen35VisionEncoder(_mmproj, ggml);
        Tensor[] stacked = encoder.EncodeVideoPairs(firsts, seconds, Side, Side);
        try
        {
            // The batch ran as one graph: a declined batch would have fallen back and said so through this flag.
            Assert.False(Field<bool>(encoder, "_batchedEncoderFailed"));
            Assert.Equal(firsts.Count, stacked.Length);

            using var managed = new Qwen35VisionEncoder(_mmproj, new CpuAllocator(BlasEnum.DotNet));
            for (int i = 0; i < stacked.Length; i++)
            {
                using Tensor alone = encoder.Encode(firsts[i], seconds[i], Side, Side);
                using Tensor reference = managed.Encode(firsts[i], seconds[i], Side, Side);
                float[] s = Values(stacked[i]), a = Values(alone), r = Values(reference);
                Assert.Equal(a.Length, s.Length);
                for (int k = 0; k < s.Length; k++)
                {
                    Assert.True(Math.Abs(s[k] - a[k]) <= 1e-5f * (1 + Math.Abs(a[k])), $"pair {i} stacked/alone differ at {k}: {s[k]} vs {a[k]}");
                    Assert.True(Math.Abs(a[k] - r[k]) <= 1e-4f * (1 + Math.Abs(r[k])), $"pair {i} GGML/managed differ at {k}: {a[k]} vs {r[k]}");
                }
            }
            // Distinct frames give distinct pairs, so equality above is not three copies of one result.
            Assert.True(MaxAbsDiff(Values(stacked[0]), Values(stacked[1])) > 1e-3f);
        }
        finally
        {
            foreach (Tensor t in stacked) t.Dispose();
        }
    }

    private static float[] Pixels(int seed)
    {
        var data = new float[3 * Side * Side];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)Math.Sin((i + 1) * 0.013 * seed + seed);
        return data;
    }

    private static float[] Values(Tensor t)
    {
        using var contiguous = t.IsContiguous() ? null : Ops.NewContiguous(t);
        return (contiguous ?? t).GetElementsAsFloat((int)t.ElementCount());
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static T Field<T>(object target, string name) => (T)target.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(target)!;
}
